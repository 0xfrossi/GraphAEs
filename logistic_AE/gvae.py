import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv, NNConv
from torch_geometric.nn import BatchNorm
from torch.nn import BatchNorm1d
from config import DEVICE as device
from utils import mat_indices

# NNConv(in_channels= encoder_embedding_size, nn=F.relu(),out_channels=encoder_embedding_size,  aggr = 'mean', )

class GVAE(nn.Module):
    def __init__(self, feature_size):
        super(GVAE, self).__init__()
        encoder_embedding_size = 47
        self.latent_embedding_size = 24
        decoder_size = 128
        edge_dim = 2
        hidden_size=36

        # Encoder layers
        self.conv1 = TransformerConv(feature_size, 
                                    encoder_embedding_size, 
                                    heads=3, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=edge_dim,
                                    dropout=0.3)
        self.bn1 = BatchNorm(encoder_embedding_size)
        self.conv2 = TransformerConv(encoder_embedding_size, 
                                    hidden_size, 
                                    heads=3, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=edge_dim,
                                    dropout=0.3)
        self.bn2 = BatchNorm(hidden_size)
        self.conv3 = TransformerConv(hidden_size, 
                                    hidden_size, 
                                    heads=3, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=edge_dim,
                                    dropout=0.3)
        self.bn3 = BatchNorm(hidden_size)

        self.zeta=  TransformerConv(hidden_size,self.latent_embedding_size,heads=3,
                                            concat=False,
                                            beta=True,
                                            edge_dim=edge_dim)

        # Latent transform layers
        self.mu_transform = TransformerConv(encoder_embedding_size, 
                                            self.latent_embedding_size,
                                            heads=3,
                                            concat=False,
                                            beta=True,
                                            edge_dim=edge_dim)
        self.logvar_transform = TransformerConv(encoder_embedding_size, 
                                            self.latent_embedding_size,
                                            heads=3,
                                            concat=False,
                                            beta=True,
                                            edge_dim=edge_dim)

        # Decoder layers
        self.decoder_dense_1 = Linear(self.latent_embedding_size*2, 64)
        self.decoder_bn_1 = BatchNorm1d(64)
        self.decoder_dense_2 = Linear(64, 64)
        self.decoder_bn_2 = BatchNorm1d(64)

        self.decoder_dense_22 = Linear(64, decoder_size)
        self.decoder_bn_22 = BatchNorm1d(decoder_size)

        #self.decoder_dense_3 = Linear(self.latent_embedding_size*2, decoder_size)
        #self.decoder_bn_3 = BatchNorm1d(decoder_size)
        self.decoder_d_3= Linear(64, decoder_size)
        self.decoder_bn_3 = BatchNorm1d(decoder_size)

        self.decoder_d_2_adj= Linear(64, decoder_size)
        self.decoder_bn_2_adj=BatchNorm1d(decoder_size)

        
        self.decoder_dense_4 = Linear(decoder_size, 1)
        self.decoder_dense_5 = Linear(decoder_size, 1)
        self.decoder_dense_adj=Linear(decoder_size, 1)

        self.apply(self._init_weights)

    def encode(self, x, edge_attr, edge_index):
        # GNN layers
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.bn1(x)
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = self.bn2(x)
        x = self.conv3(x, edge_index, edge_attr).relu()
        x = self.bn3(x)
      
        embedding=self.zeta(x, edge_index, edge_attr)

        return embedding
        

    def _init_weights(self, module):
        if  isinstance(module, Linear):
            nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def decode(self, z, batch_index):
        """
        Takes n latent vectors (one per node) and decodes them
        into adjacency matrix without main diagonal.
        """
        inputs = []

        # Iterate over graphs in batch
        for graph_id in torch.unique(batch_index):
            graph_mask = torch.eq(batch_index, graph_id)
            graph_z = z[graph_mask]

            # Get indices for adjacency matrix without main diagonal, graph_z.shape[0]= num nodi
            
            edge_indices = mat_indices(graph_z.shape[0])

            # Repeat indices to match dim of latent codes es. [1,2,3].repeat_interleave(2)-> [1,1,2,2,3,3]
            dim = self.latent_embedding_size      
            source_indices = torch.reshape(edge_indices[0].repeat_interleave(dim), (edge_indices.shape[1], dim))
            target_indices = torch.reshape(edge_indices[1].repeat_interleave(dim), (edge_indices.shape[1], dim))
            
            '''
            #dim=0
            input = torch.tensor([[1, 2,  3,  4], 
                                  [5, 6,  7,  8],
                                  [9, 10, 11, 12],
                                  [13,14, 15, 16]])
 
            indx=torch.tensor([[0 ,1, 2, 3], elem. indica il num di righe (per dim=0)
                               [3, 2, 1, 0], <-- es. 3=  riga tre pos. zero;  1=  riga uno pos. tre
                               [2, 3, 0, 1],
                               [1, 2, 1, 0]])
            #output torch.gather(input,dim=0,indx) 
            tensor([[ 1,  6, 11, 16],
                    [13, 10,  7,  4],
                    [ 9, 14,  3,  8],
                    [ 5, 10,  7,  4]])

            ##################

            torch.gather(input=input,dim= 1,index=indx) 
 
            #output
            tensor([[ 1,  2,  3,  4],
                    [ 8,  7,  6,  5],
                    [11, 12,  9, 10],
                    [14, 15, 14, 13]])

            '''
            # Gather features  graph_z= [n_nodi,16] embedding filtrato dal batch sul grafo in esame
            sources_feats = torch.gather(graph_z, 0, source_indices.to(device))
            target_feats = torch.gather(graph_z, 0, target_indices.to(device))
            sources_feats.to(device)
            target_feats.to(device)

            # Concatenate inputs of all source and target nodes
            graph_inputs = torch.cat([sources_feats, target_feats], axis=1)  #[VARIABILE,32]
            graph_inputs.to(device)
            inputs.append(graph_inputs)

        # Concatenate all inputs of all graphs in the batch
        inputs = torch.cat(inputs)
        inputs.to(device)

        # Predictions
        x = self.decoder_dense_1(inputs).relu() #[14959, 128] con x= 898
        x = self.decoder_bn_1(x)
        
        x = self.decoder_dense_2(x).relu()
        x = self.decoder_bn_2(x)
        ### end shared layers

        x_adj= self.decoder_d_2_adj(x).relu()
        x_adj = self.decoder_bn_2_adj(x_adj)
        edge_logits_adj= self.decoder_dense_adj(x_adj)

        x1 = self.decoder_dense_22(x).relu()
        x1 = self.decoder_bn_22(x1)
        edge_logits1 = self.decoder_dense_4(x1)

        x2=self.decoder_d_3(x).relu()
        x2 = self.decoder_bn_3(x2)
        edge_logits2 = self.decoder_dense_5(x2)
        #x = self.decoder_dense_3(x).relu()
        #x = self.decoder_bn_3(x)

        #[14959, 1]
        return edge_logits_adj,edge_logits1,edge_logits2

    def forward(self, x, edge_attr, edge_index, batch_index): #batch_index: array(0,0,0,0.....31,31,31) è batch.batch
                                                                                                                    
        z = self.encode(x, edge_attr, edge_index) # [num nodi batch, 16] 

        # Decode latent vector into original 
        mat_logits_adj,mat_logits1,mat_logits2 = self.decode(z, batch_index) #x=935, logits[17217, 1]---- x=1025 logits[22101, 1]

        return mat_logits_adj,mat_logits1, mat_logits2