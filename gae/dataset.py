import torch
import torch_geometric
from torch_geometric.data import Dataset,Data
import numpy as np 
import os
from tqdm import tqdm
import plan
import pickle
import json
from torch_geometric.utils import to_dense_adj
print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")
#Se si mantiene la mat. di adiacenza statica sarà necessario fare la stessa cosa con quella degli attributi dei nodi, per la convenzione
#dell'ordinamento dei nodi di torch geomentric...
#necessario avere una cartella "data" contenente 2 cartelle:
                                                            # -"processed": andrà a contenere i grafi creati dal file contenuto in "raw"
                                                            # -"raw": contiene il file con il quale si andranno a creare i grafi
class StatiDatasetPadded(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        #self.test = test
        self.filename = filename
        super(StatiDatasetPadded, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        #totale degli elementi del dataset in utilizzo
        self.lenght = 1384373      
        return [f'data_{i}.pt' for i in range(self.lenght)]

    def download(self):
        pass

    def process(self):
        #carica il file dei dati raw dalla cartella (predefinita) raw
        with open(self.raw_paths[0], "rb") as rf:
            self.piani = pickle.load(rf)
        rf.close()

        #carica dizionario di tutti i possibili nodi (es. at, truck1, depot2, ecc.) nome_nodo= "vettore one hot encoding" 
        read_file= "stati_data/dizionario_stati_nodi_f_ohe.txt"
        with open(read_file, "r") as rj:
            ohe_dict_elementi_dataset = json.load(rj)
        rj.close()

        #inizio conversione stato in grafo
        index=0
        for p in self.piani:

            #per ogni stato estraggo tutti i fatti e vado a splittare gli elementi che lo compongono (quelli che saranno i nodi)
            for stato in p.states:
                arr=[]
                lista_splittati=[]
                for elementi in stato:
                    splitted= elementi.split()
                    lista_splittati.append(splitted)
                    for e in splitted:
                        arr.append(e)

                #elimino i possibili elementi doppi
                unici= set(arr)
                # creo un dizionario per la numerazione locale del grafo, ad ogni nodo associo un numero identificativo nell'ordinamento locale dei nodi del grafo. nodo=posizione
                dict_graph_per_adiacenza=dict(zip(unici,[n for n in range(len(unici))]))
                # Stesso dizionario di quello sopra ma qui le chiavi sono le posizioni, posizione=nodo
                dict_graph_num_val=dict(zip([n for n in range(len(unici))],unici))
                    
                lista_adj=[] #lista che diventerà la matrice di adiacenza in formato sparso del grafo. Questa notazione è richiesta da pytorch geometric
                             #la lista andrà a contenere i nodi sorgente e destinazione di ogni arco, identificati dalla posizione ("ordinamento locale") contenuta in dict_graph_per_adiacenza


                #scorro tutti i fatti dello stato in esame, gestendo le due casistiche in base alla dimensione del fatto (3 o 2) per i codificare la matice di adiacenza. Avrei potuto fare 1 solo "if"...   
                #convenzione utilizzata: es. ["at truck1 depot1", "clear pallet"] at --> truck1 --> depot1 ; clear --> pallet
                for lista in lista_splittati:
                    if lista[0]=="on": 
                        lista_adj.append([ dict_graph_per_adiacenza[lista[0]] , dict_graph_per_adiacenza[lista[1]] ] )   
                        lista_adj.append([ dict_graph_per_adiacenza[lista[1]],  dict_graph_per_adiacenza[lista[2]] ] )
                    elif lista[0]=="in":
                        lista_adj.append([ dict_graph_per_adiacenza[lista[0]] , dict_graph_per_adiacenza[lista[1]] ] )   
                        lista_adj.append([ dict_graph_per_adiacenza[lista[1]],  dict_graph_per_adiacenza[lista[2]] ] )

                    elif lista[0]=="lifting":   
                        lista_adj.append([ dict_graph_per_adiacenza[lista[0]] , dict_graph_per_adiacenza[lista[1]] ] )   
                        lista_adj.append([ dict_graph_per_adiacenza[lista[1]],  dict_graph_per_adiacenza[lista[2]] ] )
                    elif lista[0]=="at":    
                        lista_adj.append([ dict_graph_per_adiacenza[lista[0]] , dict_graph_per_adiacenza[lista[1]] ] )   
                        lista_adj.append([ dict_graph_per_adiacenza[lista[1]],  dict_graph_per_adiacenza[lista[2]] ] ) 

                    elif lista[0]=="clear":
                        lista_adj.append([ dict_graph_per_adiacenza[lista[0]] , dict_graph_per_adiacenza[lista[1]] ] )

                    elif lista[0]=="available" :
                        lista_adj.append([ dict_graph_per_adiacenza[lista[0]] , dict_graph_per_adiacenza[lista[1]] ] )

            
                #creo la matice degli attributi dei nodi, cioè la codifica del tipi di nodi che appaiono nel grafo, 
                #identificati dal vettore one hot (salvato su un dizonario).
                #Ad ogni nodo (riferito dalla numerazione locale) associo il vettore one hot che lo rappresenta
                n_nodi=len(unici)
                mat_features_x=[]
                for pos,obj in dict_graph_num_val.items():  
                    for o, cod in ohe_dict_elementi_dataset.items():
                        if obj==o:           
                            mat_features_x.append(torch.LongTensor(cod)) 
                
                if len(mat_features_x)<37:             
                    while len(mat_features_x)<37:
                        mat_features_x.append(torch.LongTensor([0]*37))   

               
                #trasformo le liste di adiacenza in formato [[lista nodi sorgente],[lista nodi destinazione]] (formato richiesto torch geom.)
                l= np.array(lista_adj, dtype=np.int64) 
                cco= l.T 
                e_index_tensor=torch.from_numpy(cco)
                x_features=torch.stack(mat_features_x)  


                #creo maschera della matrice attributi nodi che esclude i nodi di padding.
                #dim. matrice nodi selezionata (senza padding): n_nodi x 37
                mask= torch.zeros(x_features.shape[0],x_features.shape[1])
                for nodo in range(n_nodi):
                    mask[nodo]=torch.ones((x_features.shape[1]))
                    
                mask=mask.type(torch.bool)

                #creo maschera della matrice di adiacenza che esclude i nodi di padding.
                #dim. matrice adiacenza selezionata (senza padding): n_nodi x n_nodi
                mask_adj= torch.zeros(x_features.shape[0],x_features.shape[1])
                for i in range(n_nodi):
                    for j in range(n_nodi):    
                        mask_adj[i][j]=1
                mask_adj=mask_adj.type(torch.bool)

                #creo una lista contenente le maschere da poter estrarre dall'elemento del batch in fase calcolo loss
                #e le posiziono nell'attributo "y" (che rimaneva vuoto).
                #potrebbe essere utile in questa fase, codificare la matrice di adiacenza (densa) flatten del grafo
                #per il confronto con quella ricostruita dal modello,
                #ed aggiungerla come terzo elemento del vettore che sarà assegnato ad "y" (in questo caso "double_mask" se si utilizzano maschere)
                #sarebbe possibile in teoria aggiungere un'ulteriore attributo all'oggetto "Data" es. "mat_adiacenza_densa", ma non so poi se/come sia
                #possibile recuperarlo dall'oggetto Batch in fase calcolo loss.
                #https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
                #Aggiungere in questa fase la matrice di adiacenza densa evita di ricalcolare la mat. del batch ed estrarle ogni volta,
                #in più si ha un maggior controllo
                double_mask=torch.stack([mask,mask_adj])


                graph= Data(x= x_features, edge_index = e_index_tensor, y=double_mask)
                torch.save(graph,os.path.join(self.processed_dir,f'data_{index}.pt'))
                index+=1                    
            
        print("processo estrazione dati completato")

    def len(self):
        #totale degli elementi del dataset in utilizzo
        return 1384373

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,f'data_{idx}.pt'))        
        return data   

