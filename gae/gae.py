import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv, NNConv,GCNConv
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn import BatchNorm, global_mean_pool
from torch_geometric.nn.aggr import Set2Set
from torch.nn import BatchNorm1d
from config import DEVICE as device

# NNConv(in_channels= encoder_embedding_size, nn=F.relu(),out_channels=encoder_embedding_size,  aggr = 'mean', )

class GVAE(nn.Module):
    def __init__(self, feature_size):
        super(GVAE, self).__init__()
        self.encoder_embedding_size = 37
        self.latent_embedding_size = 30
        #hidden_size=27
        
        # Encoder layers
        #self.conv1= GCNConv(in_channels= feature_size,out_channels= self.latent_embedding_size )
        self.conv1=TransformerConv(in_channels= feature_size,out_channels= self.latent_embedding_size*2, heads=3,concat=False,beta=False )
        self.bn1 = BatchNorm(self.latent_embedding_size*2)
        self.conv2= TransformerConv(in_channels= self.latent_embedding_size*2,out_channels= self.latent_embedding_size*2,heads=3,concat=False,beta=False)
        #self.conv2= GCNConv(in_channels= self.latent_embedding_size,out_channels= 40 )
        self.bn2 = BatchNorm(self.latent_embedding_size*2)
        self.zeta= TransformerConv(in_channels= self.latent_embedding_size*2,out_channels= self.latent_embedding_size*2,heads=3,concat=False,beta=False )
        #self.zeta=GCNConv(in_channels= 40,out_channels= 40 )
        
      
        # Decoder layers
        #self.decoder_dense_1 = Linear(self.latent_embedding_size*8, 512)
        #self.decoder_bn_1 = BatchNorm1d(512)
        #self.decoder_dense_2 = Linear(512, 512)
        #self.decoder_bn_2 = BatchNorm1d(512)
        #self.decoder_dense_shared_3 = Linear(512, 512)
        #self.decoder_bn_shared_3 = BatchNorm1d(512)

        #Adj
        self.decoder_d_first_mat_2= Linear(self.latent_embedding_size*2, 256)
        self.decoder_bn_first_mat_2=BatchNorm1d(256)
        self.decoder_d_3= Linear(256, 768)
        self.decoder_bn_3 = BatchNorm1d(768)
        self.decoder_d_4=Linear(768, 768)
        self.decoder_bn_4 = BatchNorm1d(768)
        #self.decoder_drop_mat_2=nn.Dropout(0.02)
        self.decoder_reduce_mat_2=Linear(768, 1024)
        self.decoder_bn_reduce_mat_2 = BatchNorm1d(1024)
        self.decoder_d_5=Linear(1024, 1369)
        self.decoder_bn_5=BatchNorm1d(1369)
        self.decoder_last_mat_2=Linear(1369, 1369)
        
        
        #x (self.latent_embedding_size*8, 256)
        self.decoder_d_first_mat_3=  Linear(self.latent_embedding_size*4, 256)
        self.decoder_bn_first_mat_3=BatchNorm1d(256)
        self.decoder_d_mat_3= Linear(256, 768)
        self.decoder_bn_mat_3=BatchNorm1d(768)
        self.decoder_reduce_mat_3=Linear(768, 1024)
        self.decoder_bn_reduce_mat_3 = BatchNorm1d(1024)
        self.decoder_mat_4_att= Linear(1024,1369)
        self.decoder_bn_4att= BatchNorm1d(1369)
        self.decoder_last_mat_3=Linear(1369, 1369)
        #self.out=torch.nn.Softmax(dim=1)

        self.apply(self._init_weights)

    def encode(self, x, edge_index, batch_index):
        # GNN layers
        x = self.conv1(x, edge_index).relu()
        x = self.bn1(x)
        x = self.conv2(x, edge_index ).relu()
        x = self.bn2(x)
        #x = self.conv3(x, edge_index).relu()
        #x = self.bn3(x)
      
        embedding_nodes=self.zeta(x, edge_index).relu()
        embedding= global_mean_pool(x= embedding_nodes, batch=batch_index)
        return embedding
        

    def _init_weights(self, module):
        if  isinstance(module, Linear):
            nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                #module.bias.data.zero_()
                nn.init.constant_(module.bias, 0)

    def decode_graph(self, z):
       
        # Predictions
        #x = self.decoder_dense_1(z).relu() #[14959, 128] con x= 898
        #x = self.decoder_bn_1(x)
        #x = self.decoder_dense_2(x).relu()
        #x = self.decoder_bn_2(x)
        #x=self.decoder_dense_shared_3(x).relu()
        #x=self.decoder_bn_shared_3(x) 
        
        xatt= self.decoder_d_first_mat_3(z).relu()
        xatt= self.decoder_bn_first_mat_3(xatt)
        xatt= self.decoder_d_mat_3(xatt).relu()
        xatt = self.decoder_bn_mat_3(xatt)
        xatt= self.decoder_reduce_mat_3(xatt).relu()
        xatt= self.decoder_bn_reduce_mat_3(xatt)
        xatt= self.decoder_mat_4_att(xatt).relu()
        xatt= self.decoder_bn_4att(xatt)
        xatt= self.decoder_last_mat_3(xatt)
        #xatt=self.out(xatt)

        xadj=self.decoder_d_first_mat_2(z).relu()
        xadj=self.decoder_bn_first_mat_2(xadj)
        xadj=self.decoder_d_3(xadj).relu()
        xadj=self.decoder_bn_3(xadj)
        xadj=self.decoder_d_4(xadj).relu()
        xadj=self.decoder_bn_4(xadj)
        #xadj=self.decoder_drop_mat_2(xadj)
        xadj=self.decoder_reduce_mat_2(xadj).relu()
        xadj=self.decoder_bn_reduce_mat_2(xadj)
        xadj=self.decoder_d_5(xadj).relu()
        xadj=self.decoder_bn_5(xadj)
        xadj=self.decoder_last_mat_2(xadj)
        
        return xatt, xadj


    def forward(self, x, edge_index, batch_index):                                                              
        z = self.encode(x, edge_index,batch_index )
 
        logits_nodi, logits_adj= self.decode_graph(z)
        return logits_nodi, logits_adj
