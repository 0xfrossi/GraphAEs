import torch
import torch_geometric
from torch_geometric.data import Dataset,Data
import numpy as np 
import os
from tqdm import tqdm
import plan
import action
import pickle
import json

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class DepotsDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        #self.test = test
        self.filename = filename
        super(DepotsDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.lenght = 241434      
        return [f'data_{i}.pt' for i in range(self.lenght)]

        

    def download(self):
        pass

    def process(self):

        with open(self.raw_paths[0], "rb") as rf:
            self.piani = pickle.load(rf)
        rf.close()

        read_file="/home/frossi/Codice/depots_data/dizionario_depots_azioni_f_ohe.txt"
        with open(read_file, "r") as rj:
            ohe_dict_azioni_dataset = json.load(rj)
        rj.close()

        index=0
        for p in self.piani:

            arr=[]
            lista_splittati=[]
            for elementi in p.initial_state:
                splitted= elementi.split()
                lista_splittati.append(splitted)
                for e in splitted:
                    arr.append(e)

            unici= set(arr)
            # obj=pos
            dict_graph_per_adiacenza=dict(zip(unici,[n for n in range(len(unici))]))
            # pos=obj
            dict_graph_num_val=dict(zip([n for n in range(len(unici))],unici))
                  
            lista_adj=[]
            lista_edg_f=[]
            for lista in lista_splittati:
                if len(lista)==3:
                    lista_adj.append([ dict_graph_per_adiacenza[lista[0]] , dict_graph_per_adiacenza[lista[1]] ] )
                    lista_edg_f.append(torch.tensor([1,0,0]))
                    lista_adj.append([ dict_graph_per_adiacenza[lista[1]],  dict_graph_per_adiacenza[lista[2]] ] )
                    lista_edg_f.append(torch.tensor([0,1,0]))
                else:
                    lista_adj.append([ dict_graph_per_adiacenza[lista[0]] , dict_graph_per_adiacenza[lista[1]] ] )
                    lista_edg_f.append(torch.tensor([0,0,1]))
            #obj=codifica
            #ohe_dict_azioni_dataset
            n_nodi=len(unici)
            mat_features_x=[]
            for pos,obj in dict_graph_num_val.items():  
                for o, cod in ohe_dict_azioni_dataset.items():
                    if obj==o:           
                        mat_features_x.append(torch.LongTensor(cod)) 
         
            l= np.array(lista_adj, dtype=np.int64)            
            cco= l.T
            e_index_tensor=torch.from_numpy(cco)
            x_features=torch.stack(mat_features_x)
            edge_f=torch.stack(lista_edg_f)    
                        
            graph= Data(x= x_features, edge_index = e_index_tensor, edge_attr=edge_f)
            torch.save(graph,os.path.join(self.processed_dir,f'data_{index}.pt'))
            index+=1                    
            
        print("processo estrazione dati completato")

    def len(self):
        return 241434

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,f'data_{idx}.pt'))        
        return data

   