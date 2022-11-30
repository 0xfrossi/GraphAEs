from torch_geometric.utils import to_dense_adj
import torch
import mlflow.pytorch
from rdkit import Chem
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
    # Get adjacency matrix less main diagonal for targets
    mat_indices = mat_indices(graph_targets.shape[0])
    dense_adj=to_dense_adj(mat_indices) 
    mat_mask = torch.squeeze(dense_adj).bool()
    return graph_targets[mat_mask]


def slice_graph_predictions(mat_logits, graph_size, start_point):
    """
    Slices out the corresponding section from a list of batch mat values.
    Given a start point and the size of a graph's, simply slices
    the section from the batch list.
    -------
    mat_logits: A batch of mat predictions of different graphs
    graph_size: Size of the graph to slice
    start_point: Index of the first node of this graph
    """
    graph_logits = torch.squeeze(mat_logits[start_point:start_point + graph_size])  
    return graph_logits



def check_graph_reconstruction(graph_predictions, graph_targets):
    """
    Checks if the adjacency matrix prediction matches the ground-truth of the graph 
    """
    # Apply sigmoid to get binary prediction values
    preds = (torch.sigmoid(graph_predictions.view(-1)) > 0.5).int()
    # Reshape the targets
    labels = graph_targets.view(-1)
    # Check if the predictions and the groundtruth match
    if labels.shape[0] == sum(torch.eq(preds, labels)): #se preds e lab sn uguali ritorna un vett di "true"*dim(labs)
        return True
    return False    

def gvae_loss(mat_logits, edge_index,  batch_index):
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


def reconstruction_accuracy(mat_logits, edge_index, batch_index):
    # Convert edge index to adjacency matrix
    batch_targets = torch.squeeze(to_dense_adj(edge_index))
    # Store target mats
    batch_targets_mats = []
    # Iterate over batch and collect each of the mats
    batch_node_counter = 0
    num_recon = 0
    for graph_id in torch.unique(batch_index):
        
        graph_targets_mat = slice_graph_targets(graph_id, 
                                                batch_targets, 
                                                batch_index)
        graph_predictions_mat = slice_graph_predictions(mat_logits, 
                                                        graph_targets_mat.shape[0], 
                                                        batch_node_counter)

        # Update counter to the index of the next graph
        batch_node_counter = batch_node_counter + graph_targets_mat.shape[0]

        # Slice node features of this batch
        #graph_node_features = slice_node_features(graph_id, node_features, batch_index)
        graph_node_features=None
        # Check if graph is successfully reconstructed
        num_nodes = sum(torch.eq(batch_index, graph_id))
        recon = check_graph_reconstruction(graph_predictions_mat,graph_targets_mat,graph_node_features, num_nodes) 
        num_recon = num_recon + int(recon)

        # Add targets to mat list
        batch_targets_mats.append(graph_targets_mat)
        
    # Calculate accuracy between predictions and labels at batch level
    batch_targets_mats = torch.cat(batch_targets_mats)  
    mat_discrete = torch.squeeze(torch.tensor(torch.sigmoid(mat_logits) > 0.5, dtype=torch.int32))
    acc = torch.true_divide(torch.sum(batch_targets_mats==mat_discrete), batch_targets_mats.shape[0]) 

    return acc, num_recon


def mat_indices(dim):
    src=[]
    tgt=[]
    for s in range(dim):
        for t in range(dim):
            if s!=t:
                src.append(s)
                tgt.append(t)
    ts=torch.tensor(src,dtype=torch.int64)  
    tt=torch.tensor(tgt,dtype=torch.int64) 
    indicies=torch.stack([ts,tt],dim=0)
    return indicies   