import torch
from torch_geometric.loader import DataLoader
from logistic_data.dataset import LogisticDataset
from tqdm import tqdm
import numpy as np
from utils import count_parameters, gvae_loss, reconstruction_accuracy, slice_edge_type_from_edge_feats
from gvae import GVAE
from config import DEVICE as device

batch_dim=10
# Load data
print("Inizio caricamento dati...")
train_data= LogisticDataset(root="logistic_data/data/", filename="logistics_plans")[:100]
test_data= LogisticDataset(root="logistic_data/data/", filename="logistics_plans")[100:150]
final_test_data=LogisticDataset(root="logistic_data/data/", filename="logistics_plans")[200:250]

train_data.shuffle()
test_data.shuffle()

train_loader = DataLoader(train_data, batch_size=batch_dim, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_dim, shuffle=True)
final_test_loader = DataLoader(test_data, batch_size=batch_dim, shuffle=True)

print("Fine caricamento dati...")

# Load model
model = GVAE(feature_size=train_data[0].x.shape[1])
model = model.to(device)
print("Model parameters: ", count_parameters(model))

# Define loss and optimizer
loss_fn = gvae_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.009, weight_decay=1e-6)

# Train function
def run_one_epoch(data_loader, type, epoch):
    # Store per batch loss and accuracy 
    
    all_losses = []
    all_accs_0 = []
    all_accs_1 = []
    all_accs_adj = []
    # Save some numbers
    total_graph = 0
    reconstructed_graph_0 = 0
    reconstructed_graph_1 = 0
    reconstructed_graph_adj = 0

    # Iterate over data loader
    for i, batch in enumerate(tqdm(data_loader)):
       
           
        batch=batch.to(device)  
        # Reset gradients
        optimizer.zero_grad() 
        # Call model
        mat_logits_adj,mat_logits_0, mat_logits_1 = model(batch.x.float(), batch.edge_attr.float(),batch.edge_index,batch.batch) 
        discrete_edge_attr=slice_edge_type_from_edge_feats(batch.edge_attr.float())
        # Calculate loss and backpropagate
        loss = loss_fn(mat_logits_adj,mat_logits_0,mat_logits_1, batch.edge_index, batch.batch, discrete_edge_attr )
        if type == "Train":

            loss.backward()  
            optimizer.step() 

        # Calculate metrics
        acc_adj,acc_0,acc_1, num_recon_0, num_recon_1,num_recon_adj = reconstruction_accuracy(mat_logits_adj,mat_logits_0, mat_logits_1,batch.edge_index, batch.batch,discrete_edge_attr)
        total_graph = total_graph + batch.num_graphs

        reconstructed_graph_0 = reconstructed_graph_0 + num_recon_0
        reconstructed_graph_1 = reconstructed_graph_1 + num_recon_1
        reconstructed_graph_adj= reconstructed_graph_adj + num_recon_adj
        # Store loss and metrics
        all_losses.append(loss.detach().cpu().numpy())
        all_accs_0.append(acc_0.detach().cpu().numpy())
        all_accs_1.append(acc_1.detach().cpu().numpy())
        all_accs_adj.append(acc_adj.detach().cpu().numpy())


    print(f"{type} epoch {epoch+1} loss: ", np.array(all_losses).mean())
    
    print(f"{type} epoch {epoch+1} accuracy adj mat. ': ", np.array(all_accs_adj).mean())
    print(f"{type} epoch {epoch+1} accuracy mat. type 'at1': ", np.array(all_accs_0).mean())
    print(f"{type} epoch {epoch+1} accuracy mat. type 'at2': ", np.array(all_accs_1).mean())

    print(f" Adj mat. Reconstructed {reconstructed_graph_adj} out of {total_graph} graphs.")
    print(f" Type 'at1' Reconstructed {reconstructed_graph_0} out of {total_graph} graphs.")
    print(f" Type 'at2' Reconstructed {reconstructed_graph_1} out of {total_graph} graphs.")
    print("\n****###**###**###****\n")

# Run training

n_epoch=21
for epoch in range(n_epoch): 
    model.train()
    run_one_epoch(train_loader, type="Train", epoch=epoch)
    if epoch % 4 == 0:
        print("Start test epoch...")
        model.eval()
        run_one_epoch(test_loader, type="Test", epoch=epoch)
    if epoch==n_epoch-1:
        print("Final test epoch...")
        model.eval()
        run_one_epoch(final_test_loader, type="Test", epoch=epoch)