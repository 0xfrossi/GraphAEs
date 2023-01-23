from torch_geometric.utils import to_dense_adj
import torch
#import mlflow.pytorch
#from rdkit import Chem
from config import DEVICE as device

def count_parameters(model):
    """
    Counts the number of parameters for a Pytorch model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def slice_graph_targets(graph_id, batch_targets, batch_index):
    """
    Slices out the  adjacency matrix without main diagonal for
    a single graph from a large adjacency matrix for a full batch.
    --------
    graph_id: The ID of the graph (in the batch index) to slice
    batch_targets: A dense adjacency matrix for the whole batch
    batch_index: The node to graph map for the batch
    """
    # Create mask for nodes of this graph id
    graph_mask = torch.eq(batch_index, graph_id)
    # Row slice and column slice batch targets to get graph targets
    graph_targets = batch_targets[graph_mask][:, graph_mask]
    # Get adjacency matrix 
    ind = mat_indices(graph_targets.shape[0])
    dense_adj=to_dense_adj(ind) 
    mat_mask = torch.squeeze(dense_adj).bool()
    return graph_targets[mat_mask]


def slice_graph_predictions(mat_logits_0,mat_logits_1, mat_logits_2,graph_size, start_point):
    """
    Slices out the corresponding section from a list of batch mat values.
    Given a start point and the size of a graph's, simply slices
    the section from the batch list.
    -------
    mat_logits: A batch of mat predictions of different graphs
    graph_size: Size of the graph to slice
    start_point: Index of the first node of this graph
    """
    graph_logits_0 = torch.squeeze(mat_logits_0[start_point:start_point + graph_size])  
    graph_logits_1 = torch.squeeze(mat_logits_1[start_point:start_point + graph_size])  
    graph_logits_2 = torch.squeeze(mat_logits_2[start_point:start_point + graph_size]) 
    #return graph_logits_0, graph_logits_1, graph_logits_adj
    return graph_logits_0, graph_logits_1,graph_logits_2




def check_graph_reconstruction(graph_predictions, graph_targets):
    """
    Checks if the adjacency matrix prediction matches the ground-truth of the graph 
    """
    # Apply sigmoid to get binary prediction values
    preds = (torch.sigmoid(graph_predictions.view(-1)) > 0.5).int()
    # Reshape the targets
    labels = graph_targets.reshape(-1)
    # Check if the predictions and the groundtruth match
    if labels.shape[0] == sum(torch.eq(preds, labels)): #se preds e lab sn uguali ritorna un vett di "true"*dim(labs)
        return True
    return False    

def gvae_loss(mat_logits_adj,mat_logits_0, mat_logits_1, edge_index,  batch_index, discrete_edge_att):
    """
    Calculates a weighted ELBO loss for a batch of graphs for the graph autoencoder model.
    """
    # Convert target edge index to dense adjacency matrix
    batch_targets = torch.squeeze(to_dense_adj(edge_index))
    batch_targets_types=torch.squeeze(to_dense_adj(edge_index=edge_index,edge_attr=discrete_edge_att))
    # Reconstruction loss per graph
    batch_recon_loss = []
    batch_node_counter = 0

    # Loop over graphs in this batch
    for graph_id in torch.unique(batch_index):
        # Get targets for this graph from the whole batch

        graph_targets_adj= slice_graph_targets(graph_id, 
                                                batch_targets, 
                                                batch_index)

        graph_targets_mat = slice_graph_targets(graph_id, 
                                                batch_targets_types, 
                                                batch_index)

        # Get predictions for this graph from the whole batch
        graph_predictions_mat_0,graph_predictions_mat_1, graph_prediction_adj = slice_graph_predictions(mat_logits_adj, mat_logits_0, 
                                                        mat_logits_1,
                                                        graph_targets_mat.shape[0], #flatten matrix  
                                                        batch_node_counter)
        
        # Update counter to the index of the next graph
        batch_node_counter = batch_node_counter + graph_targets_mat.shape[0]

        target_mats=adj_to_3d(graph_targets_mat,2)
        target_mat_0=target_mats[0]
        target_mat_1=target_mats[1]
        # Calculate edge-weighted binary cross entropy
        # adj mat. 
        weight_a = graph_targets_mat.shape[0]/sum(graph_targets_adj)
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=weight_a).to(device)
        graph_recon_loss_a = bce(graph_prediction_adj.view(-1), graph_targets_adj.view(-1))
        batch_recon_loss.append(graph_recon_loss_a) 

        #first mat. type
        weight = graph_targets_mat.shape[0]/sum(target_mat_0)
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=weight).to(device)
        graph_recon_loss0 = bce(graph_predictions_mat_0.view(-1), target_mat_0.reshape(-1))
        batch_recon_loss.append(graph_recon_loss0) 

        #Second mat. type
        weight_1 = graph_targets_mat.shape[0]/sum(target_mat_1)
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=weight_1).to(device)
        graph_recon_loss1 = bce(graph_predictions_mat_1.view(-1), target_mat_1.reshape(-1))
        batch_recon_loss.append(graph_recon_loss1)   

    # Take average of all losses
    num_graphs = torch.unique(batch_index).shape[0]
    batch_recon_loss = sum(batch_recon_loss) / (3*num_graphs)
    
    return batch_recon_loss 

def gvae_loss_and_rec_accuracy(mat_logits_0, mat_logits_1, mat_logits_2,edge_index,  batch_index, discrete_edge_att):
    """
    Calculates a weighted ELBO loss for a batch of graphs for the graph autoencoder model.
    """
    # Convert target edge index to dense adjacency matrix
    batch_targets = torch.squeeze(to_dense_adj(edge_index))
    batch_targets_types=torch.squeeze(to_dense_adj(edge_index=edge_index,edge_attr=discrete_edge_att))
    # Reconstruction loss per graph
    batch_recon_loss = []
    batch_node_counter = 0

    # Store target mats
    batch_targets_mats_adj = []
    batch_targets_mats_0 = []
    batch_targets_mats_1 = []
    batch_targets_mats_2 = []

    # Iterate over batch and collect each of the mats
    #batch_node_counter = 0
    num_recon_adj=0
    num_recon_0 = 0
    num_recon_1=0
    num_recon_2=0

    graph_rec_complete=0

    # Loop over graphs in this batch
    for graph_id in torch.unique(batch_index):
        # Get targets for this graph from the whole batch

        graph_targets_adj= slice_graph_targets(graph_id, 
                                                batch_targets, 
                                                batch_index)

        graph_targets_mat = slice_graph_targets(graph_id, 
                                                batch_targets_types, 
                                                batch_index)

        # Get predictions for this graph from the whole batch
        graph_predictions_mat_0,graph_predictions_mat_1, graph_predictions_mat_2 = slice_graph_predictions( mat_logits_0, 
                                                        mat_logits_1,
                                                        mat_logits_2,
                                                        graph_targets_mat.shape[0], #flatten matrix  
                                                        batch_node_counter)
        
        # Update counter to the index of the next graph
        batch_node_counter = batch_node_counter + graph_targets_mat.shape[0]

        target_mats=adj_to_3d(graph_targets_mat,3)
        target_mat_0=target_mats[0]
        target_mat_1=target_mats[1]
        target_mat_2=target_mats[2]
        # Calculate edge-weighted binary cross entropy
        # adj mat. 
        #weight_a = graph_targets_mat.shape[0]/sum(graph_targets_adj)
        #bce = torch.nn.BCEWithLogitsLoss(pos_weight=weight_a).to(device)
        #graph_recon_loss_a = bce(graph_prediction_adj.view(-1), graph_targets_adj.view(-1))
        #batch_recon_loss.append(graph_recon_loss_a) 

        #first mat. type
        weight = graph_targets_mat.shape[0]/sum(target_mat_0)
        bce_0 = torch.nn.BCEWithLogitsLoss(pos_weight=weight).to(device)
        graph_recon_loss0 = bce_0(graph_predictions_mat_0.view(-1), target_mat_0.view(-1))
        batch_recon_loss.append(graph_recon_loss0) 

        #Second mat. type
        weight_1 = graph_targets_mat.shape[0]/sum(target_mat_1)
        bce_1 = torch.nn.BCEWithLogitsLoss(pos_weight=weight_1).to(device)
        graph_recon_loss1 = bce_1(graph_predictions_mat_1.view(-1), target_mat_1.view(-1))
        batch_recon_loss.append(graph_recon_loss1)   

        #Thirth mat. type
        weight_2 = graph_targets_mat.shape[0]/sum(target_mat_2)
        bce_2 = torch.nn.BCEWithLogitsLoss(pos_weight=weight_2).to(device)
        graph_recon_loss2 = bce_2(graph_predictions_mat_2.view(-1), target_mat_2.view(-1))
        batch_recon_loss.append(graph_recon_loss2)

        #Sum type's= adj, sum_prediction=adj
        weight_sum = graph_targets_mat.shape[0]/sum(graph_targets_adj)
        bce_adj = torch.nn.BCEWithLogitsLoss(pos_weight=weight_sum).to(device)

        sum_pred_01=torch.add(graph_predictions_mat_0,graph_predictions_mat_1)
        sum_prediction= torch.add(sum_pred_01,graph_predictions_mat_2)

        graph_recon_loss_sum = bce_adj(sum_prediction.view(-1), graph_targets_adj.view(-1))
        batch_recon_loss.append(graph_recon_loss_sum)


        #####################################
        ##### Recostruction eval Part #######
        #####################################

        # Check if graph is successfully reconstructed adj
        recon_adj = check_graph_reconstruction(sum_prediction, graph_targets_adj) 
        num_recon_adj = num_recon_adj + int(recon_adj)

        # Check if graph is successfully reconstructed 0
        recon_0 = check_graph_reconstruction(graph_predictions_mat_0,target_mat_0) 
        num_recon_0 = num_recon_0 + int(recon_0)

        # Check if graph is successfully reconstructed 1
        recon_1 = check_graph_reconstruction(graph_predictions_mat_1,target_mat_1) 
        num_recon_1 = num_recon_1 + int(recon_1)

        # Check if graph is successfully reconstructed 2
        recon_2 = check_graph_reconstruction(graph_predictions_mat_2,target_mat_2) 
        num_recon_2 = num_recon_2 + int(recon_2)

        if (int(recon_adj)+int(recon_0)+int(recon_1)+int(recon_2))==4:
            graph_rec_complete+=1

        # Add targets to mat list
        batch_targets_mats_adj.append(graph_targets_adj)
        batch_targets_mats_0.append(target_mat_0)
        batch_targets_mats_1.append(target_mat_1)
        batch_targets_mats_2.append(target_mat_2)

    # Calculate accuracy between predictions and labels at batch level
    batch_targets_mats_adj=torch.cat(batch_targets_mats_adj) 
    batch_targets_mats_0 = torch.cat(batch_targets_mats_0) 
    batch_targets_mats_1 = torch.cat(batch_targets_mats_1)
    batch_targets_mats_2 = torch.cat(batch_targets_mats_2) 

    #adj
    sum_01=torch.add(mat_logits_0,mat_logits_1)
    mat_discrete_adj = torch.squeeze(torch.tensor(torch.sigmoid(torch.add(sum_01,mat_logits_2)) > 0.5, dtype=torch.int32))
    acc_adj = torch.true_divide(torch.sum(batch_targets_mats_adj==mat_discrete_adj), batch_targets_mats_adj.shape[0]) 

    #type 0
    mat_discrete_0 = torch.squeeze(torch.tensor(torch.sigmoid(mat_logits_0) > 0.5, dtype=torch.int32))
    acc_0 = torch.true_divide(torch.sum(batch_targets_mats_0==mat_discrete_0), batch_targets_mats_0.shape[0]) 

    #type 1
    mat_discrete_1 = torch.squeeze(torch.tensor(torch.sigmoid(mat_logits_1) > 0.5, dtype=torch.int32))
    acc_1 = torch.true_divide(torch.sum(batch_targets_mats_1==mat_discrete_1), batch_targets_mats_1.shape[0])

    #type 2
    mat_discrete_2 = torch.squeeze(torch.tensor(torch.sigmoid(mat_logits_2) > 0.5, dtype=torch.int32))
    acc_2 = torch.true_divide(torch.sum(batch_targets_mats_2==mat_discrete_2), batch_targets_mats_2.shape[0])


    # Take average of all losses
    num_graphs = torch.unique(batch_index).shape[0]
    batch_recon_loss = sum(batch_recon_loss) / (4*num_graphs)
    
    return batch_recon_loss, acc_adj, acc_0,acc_1, acc_2, num_recon_0, num_recon_1,num_recon_2, num_recon_adj,graph_rec_complete





def reconstruction_accuracy(mat_logits_0,mat_logits_1,mat_logits_2 ,edge_index, batch_index,discrete_edge_att):
    # Convert edge index to adjacency matrix
    batch_targets = torch.squeeze(to_dense_adj(edge_index))
    batch_targets_types=torch.squeeze(to_dense_adj(edge_index=edge_index,edge_attr=discrete_edge_att))

    # Store target mats
    batch_targets_mats_adj = []
    batch_targets_mats_0 = []
    batch_targets_mats_1 = []
    batch_targets_mats_2 = []

    # Iterate over batch and collect each of the mats
    batch_node_counter = 0
    num_recon_adj=0
    num_recon_0 = 0
    num_recon_1=0
    num_recon_2=0

    graph_rec_complete=0

    for graph_id in torch.unique(batch_index):
        
        graph_targets_adj= slice_graph_targets(graph_id, 
                                                batch_targets, 
                                                batch_index)

        graph_targets_mat = slice_graph_targets(graph_id, 
                                                batch_targets_types, 
                                                batch_index)
                                                
        graph_predictions_mat_0, graph_predictions_mat_1,graph_predictions_mat_2 = slice_graph_predictions(mat_logits_0, 
                                                        mat_logits_1,
                                                        mat_logits_2,
                                                        graph_targets_mat.shape[0], 
                                                        batch_node_counter)

        # Update counter to the index of the next graph
        batch_node_counter = batch_node_counter + graph_targets_mat.shape[0]

        # Slice node features of this batch
        
        target_mats=adj_to_3d(graph_targets_mat,3)
        target_mat_0=target_mats[0]
        target_mat_1=target_mats[1]
        target_mat_2=target_mats[2]

        sum_01=torch.add(graph_predictions_mat_0,graph_predictions_mat_1)
        graph_prediction_adj=torch.add(sum_01,graph_predictions_mat_2)

        # Check if graph is successfully reconstructed adj
        recon_adj = check_graph_reconstruction(graph_prediction_adj, graph_targets_adj) 
        num_recon_adj = num_recon_adj + int(recon_adj)

        # Check if graph is successfully reconstructed 0
        recon_0 = check_graph_reconstruction(graph_predictions_mat_0,target_mat_0) 
        num_recon_0 = num_recon_0 + int(recon_0)

        # Check if graph is successfully reconstructed 1
        recon_1 = check_graph_reconstruction(graph_predictions_mat_1,target_mat_1) 
        num_recon_1 = num_recon_1 + int(recon_1)

        # Check if graph is successfully reconstructed 2
        recon_2 = check_graph_reconstruction(graph_predictions_mat_2,target_mat_2) 
        num_recon_2 = num_recon_2 + int(recon_2)

        if (int(recon_adj)+int(recon_0)+int(recon_1)+int(recon_2))==4:
            graph_rec_complete+=1

        # Add targets to mat list
        batch_targets_mats_adj.append(graph_targets_adj)
        batch_targets_mats_0.append(target_mat_0)
        batch_targets_mats_1.append(target_mat_1)
        batch_targets_mats_2.append(target_mat_2)

    # Calculate accuracy between predictions and labels at batch level
    batch_targets_mats_adj = torch.cat(batch_targets_mats_adj) 
    batch_targets_mats_0 = torch.cat(batch_targets_mats_0) 
    batch_targets_mats_1 = torch.cat(batch_targets_mats_1) 
    batch_targets_mats_2 = torch.cat(batch_targets_mats_2) 

    #adj
    sum_01=torch.add(mat_logits_0,mat_logits_1)
    mat_discrete_adj = torch.squeeze(torch.tensor(torch.sigmoid(torch.add(sum_01,mat_logits_2)) > 0.5, dtype=torch.int32))
    acc_adj = torch.true_divide(torch.sum(batch_targets_mats_adj==mat_discrete_adj), batch_targets_mats_adj.shape[0]) 

    #type 0
    mat_discrete_0 = torch.squeeze(torch.tensor(torch.sigmoid(mat_logits_0) > 0.5, dtype=torch.int32))
    acc_0 = torch.true_divide(torch.sum(batch_targets_mats_0==mat_discrete_0), batch_targets_mats_0.shape[0]) 

    #type 1
    mat_discrete_1 = torch.squeeze(torch.tensor(torch.sigmoid(mat_logits_1) > 0.5, dtype=torch.int32))
    acc_1 = torch.true_divide(torch.sum(batch_targets_mats_1==mat_discrete_1), batch_targets_mats_1.shape[0])

    mat_discrete_2 = torch.squeeze(torch.tensor(torch.sigmoid(mat_logits_2) > 0.5, dtype=torch.int32))
    acc_2 = torch.true_divide(torch.sum(batch_targets_mats_2==mat_discrete_2), batch_targets_mats_2.shape[0])

    return acc_adj, acc_0,acc_1, acc_2, num_recon_0, num_recon_1, num_recon_2,num_recon_adj,graph_rec_complete


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

def slice_edge_type_from_edge_feats(edge_feats):
    """
    This function return a vector of discrete edge's types instead of one-hot-enc.
    rappresentation
    """
    edge_types_one_hot = edge_feats[:, :3]
    edge_types = edge_types_one_hot.nonzero(as_tuple=False)
    # Start index at 1, zero will be no edge
    edge_types[:, 1] = edge_types[:, 1] + 1
    return edge_types[:,1]  


def adj_to_3d(matrix,deep):
    """
    Creates a multi-dim tensor with dim: (n_edge_type,n_nodes,n_nodes)
    each matrix is the adj matrix estimate for only 1 edge's type.
    """   
    mat_3d=torch.zeros(deep,matrix.shape[0])
    for tipo in range(deep):
        mat_3d[tipo,:] = torch.eq(matrix,tipo+1).type(torch.DoubleTensor)

    return  mat_3d      



def gvae_loss_edge(mat_logits, edge_index,  batch_index):
    """
    Calculates a weighted ELBO loss for a batch of graphs for the graph autoencoder model.
    """
    # Convert target edge index to dense adjacency matrix
    batch_targets = torch.squeeze(to_dense_adj(edge_index))

    # Reconstruction loss per graph
    batch_recon_loss = []
    batch_node_counter = 0

    # Loop over graphs in this batch
    for graph_id in torch.unique(batch_index):
        # Get targets for this graph from the whole batch
        graph_targets_mat = slice_graph_targets(graph_id, 
                                                batch_targets, 
                                                batch_index)

        # Get predictions for this graph from the whole batch
        graph_predictions_mat = slice_graph_predictions(mat_logits, 
                                                        graph_targets_mat.shape[0], #flatten matrix meno la diag 
                                                        batch_node_counter)
        
        # Update counter to the index of the next graph
        batch_node_counter = batch_node_counter + graph_targets_mat.shape[0]

        # Calculate edge-weighted binary cross entropy
        weight = graph_targets_mat.shape[0]/sum(graph_targets_mat)
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=weight).to(device)
        graph_recon_loss = bce(graph_predictions_mat.view(-1), graph_targets_mat.view(-1))
        batch_recon_loss.append(graph_recon_loss)   

    # Take average of all losses
    num_graphs = torch.unique(batch_index).shape[0]
    batch_recon_loss = sum(batch_recon_loss) / num_graphs
    
    return batch_recon_loss       