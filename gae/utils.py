from torch_geometric.utils import to_dense_adj
import torch
from sklearn.metrics import accuracy_score,precision_score,recall_score
from config import DEVICE as device

def count_parameters(model):
    """
    Counts the number of parameters for a Pytorch model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#estrae dalla matrice di adiacenza del batch la matrice di adiacenza del grafo selezionato
#se la matice di adiacenza densa è salvata nell' oggetto grafo, questo metodo non serve
def slice_graph_targets(graph_id, batch_targets, batch_index):
  
    # Create mask for nodes of this graph id
    graph_mask = torch.eq(batch_index, graph_id)
    
    # Row slice and column slice batch targets to get graph targets
    graph_targets = batch_targets[graph_mask][:, graph_mask]
    # Get adjacency matrix 
    ind = mat_indices(37)
    dense_adj=to_dense_adj(ind, max_num_nodes=37) 
    mat_mask = torch.squeeze(dense_adj).bool()
    return graph_targets[mat_mask]


def is_tensors_equal(tensor1, tensor2):
    return torch.eq(tensor1, tensor2).all().item() == 1

def gvae_loss_and_rec_accuracy( mat_logits_att, mat_logits_adj, edge_index,  batch_index, batch_x, batch_y):
    num_graphs = torch.unique(batch_index).shape[0]

    # Convert target edge index to dense adjacency matrix
    batch_targets = torch.squeeze(to_dense_adj(edge_index, max_num_nodes=37*num_graphs))
    

    #liste che conterranno i valori delle metriche 
    batch_recon_loss = []
    batch_att_loss=[]
    batch_adj_loss=[]
    batch_acc_adj=[]
    batch_acc_att=[]
    batch_pre_adj=[]
    batch_pre_att=[]
    batch_rec_adj=[]
    batch_rec_att=[]
    
    num_recon_adj=0
    num_recon_att = 0
    graph_rec_complete=0
    graph_adj_t=[] #contiene le mat. dense di adiacenza di tutti i grafi del batch

    # per ogni grafo del batch estraggo, dalla matrice batch_targets, la matice densa di adiacenza di ogni grafo e la aggiungo in "graph_adj_t"

    for graph_id in torch.unique(batch_index):
        # Get targets for this graph from the whole batch
        graph_targets_adj = slice_graph_targets(graph_id, batch_targets, batch_index)
        graph_targets_adj=graph_targets_adj.reshape(-1)
        graph_adj_t.append(graph_targets_adj)
    graph_adj_t=torch.stack(graph_adj_t)
    


    graph_adj_t=graph_adj_t.reshape(-1,37,37)    
    batch_x=batch_x.reshape(-1,37,37)
    batch_y=batch_y.reshape(-1,2,37,37)
    mat_logits_att=mat_logits_att.reshape(-1,37,37)
    mat_logits_adj=mat_logits_adj.reshape(-1,37,37)


    #bce_cat=torch.nn.CrossEntropyLoss(reduction='none').to(device)
    bce_x = torch.nn.BCEWithLogitsLoss().to(device)
    bce = torch.nn.BCEWithLogitsLoss().to(device)
    
    #per ogni grafo nel batch vado ad estrarre le maschere corrispondenti, le applico e vado a calcolare loss e metriche
    for i in range(batch_x.shape[0]):

        masked_x_t=batch_x[i][batch_y[i][0]]  
        masked_logits_att=mat_logits_att[i][batch_y[i][0]]
        masked_logits_adj=mat_logits_adj[i][batch_y[i][1]]

        masked_adj_t=graph_adj_t[i][batch_y[i][1]]

        graph_recon_loss_att = bce_x(masked_logits_att.view(-1), masked_x_t.view(-1)) 
        batch_att_loss.append(graph_recon_loss_att)

        graph_recon_loss_a = bce(masked_logits_adj, masked_adj_t)
        batch_adj_loss.append(graph_recon_loss_a)


        mat_discrete_adj = (torch.sigmoid(masked_logits_adj.view(-1)) > 0.50).type(torch.int32)
        #attributi nodi
        mat_discrete_att = (torch.sigmoid(masked_logits_att.view(-1)) > 0.50).type(torch.int32)

        # eval accuracy 
        acc_vect_adj= is_tensors_equal(masked_adj_t,mat_discrete_adj)
        num_recon_adj=num_recon_adj + int(acc_vect_adj)
        acc_vect_x= is_tensors_equal(masked_x_t.view(-1),mat_discrete_att)
        num_recon_att= num_recon_att + int(acc_vect_x)
        if int(acc_vect_adj) + int(acc_vect_x)==2:
            graph_rec_complete+=1    

        #calcolo metriche
        acc_adj=accuracy_score(masked_adj_t.detach().cpu().numpy(),mat_discrete_adj.detach().cpu().numpy())
        batch_acc_adj.append(acc_adj)
        acc_att=accuracy_score(masked_x_t.view(-1).detach().cpu().numpy(),mat_discrete_att.detach().cpu().numpy())
        batch_acc_att.append(acc_att)
        pre_adj=precision_score(masked_adj_t.detach().cpu().numpy(),mat_discrete_adj.detach().cpu().numpy(),zero_division=0,average="binary")
        batch_pre_adj.append(pre_adj)
        pre_att=precision_score(masked_x_t.view(-1).detach().cpu().numpy(),mat_discrete_att.detach().cpu().numpy(),zero_division=0,average="binary")
        batch_pre_att.append(pre_att)
        rec_adj=recall_score(masked_adj_t.detach().cpu().numpy(),mat_discrete_adj.detach().cpu().numpy(),zero_division=0,average="binary")
        batch_rec_adj.append(rec_adj)
        rec_att=recall_score(masked_x_t.view(-1).detach().cpu().numpy(),mat_discrete_att.detach().cpu().numpy(),zero_division=0,average="binary")
        batch_rec_att.append(rec_att)

    #calcolo valori medi del batch
    batch_att_loss=sum(batch_att_loss)
    batch_adj_loss=sum(batch_adj_loss)
    batch_recon_loss.append(batch_adj_loss)
    batch_recon_loss.append(batch_att_loss)
    batch_recon_loss=sum(batch_recon_loss)

    fin_acc_adj=sum(batch_acc_adj)/len(batch_acc_adj)
    fin_acc_att=sum(batch_acc_att)/len(batch_acc_att)
    fin_pre_adj=sum(batch_pre_adj)/len(batch_pre_adj)
    fin_pre_att=sum(batch_pre_att)/len(batch_pre_att)
    fin_rec_adj=sum(batch_rec_adj)/len(batch_rec_adj)
    fin_rec_att=sum(batch_rec_att)/len(batch_rec_att)
    score=[fin_pre_adj,fin_pre_att,fin_rec_adj,fin_rec_att]

    #imposto a zero alcune metriche che non mi servono. Vengono restituite comunque ma non usate
    acc_batch_adj=torch.tensor([0])
    acc_batch_att=torch.tensor([0])
    return batch_recon_loss, fin_acc_adj, fin_acc_att, num_recon_att, num_recon_adj,graph_rec_complete,score,acc_batch_adj, acc_batch_att


#costriusce una matrice quadrata piena, nel formato pytorch geometric.
def mat_indices(dim):
    src=[]
    tgt=[]
    for s in range(dim):
        for t in range(dim):
            src.append(s)
            tgt.append(t)

    ts=torch.tensor(src,dtype=torch.int64)  
    tt=torch.tensor(tgt,dtype=torch.int64) 
    indicies=torch.stack([ts,tt],dim=0)
    return indicies   
