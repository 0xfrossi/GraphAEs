import torch
from torch_geometric.loader import DataLoader
from logistic_data.dataset import LogisticDataset
from tqdm import tqdm
import numpy as np
from utils import count_parameters, gvae_loss, reconstruction_accuracy
from gvae import GVAE
from config import DEVICE as device

batch_dim=200
# Load data
train_data= LogisticDataset(root="logistic_data/data/", filename="logistics_plans")[:20000]
test_data= LogisticDataset(root="logistic_data/data/", filename="logistics_plans")[20000:25000]
final_test_data=LogisticDataset(root="logistic_data/data/", filename="logistics_plans")[100000:110000]

train_data.shuffle()
test_data.shuffle()

train_loader = DataLoader(train_data, batch_size=batch_dim, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_dim, shuffle=True)
final_test_loader = DataLoader(test_data, batch_size=batch_dim, shuffle=True)

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
    all_accs = []
    # Save some numbers
    total_graph = 0
    reconstructed_graph = 0

    # Iterate over data loader
    for i, batch in enumerate(tqdm(data_loader)):
       
        try:
           
            batch=batch.to(device)  
            # Reset gradients
            optimizer.zero_grad() 
            # Call model
            mat_logits = model(batch.x.float(), batch.edge_attr.float(),batch.edge_index,batch.batch) 
            # Calculate loss and backpropagate
            loss = loss_fn(mat_logits, batch.edge_index, batch.batch )
            if type == "Train":

                loss.backward()  
                optimizer.step() 

            # Calculate metrics
            acc, num_recon = reconstruction_accuracy(mat_logits, batch.edge_index, batch.batch, batch.x.float())
            total_graph = total_graph + batch.num_graphs
            reconstructed_graph = reconstructed_graph + num_recon

            # Store loss and metrics
            all_losses.append(loss.detach().cpu().numpy())
            all_accs.append(acc.detach().cpu().numpy())
           # all_kldivs.append(kl_div.detach().cpu().numpy())
           

        except IndexError as error:
            print("Error: ", error)
    
    print(f"{type} epoch {epoch+1} loss: ", np.array(all_losses).mean())
    print(f"{type} epoch {epoch+1} accuracy: ", np.array(all_accs).mean())
    print(f"Reconstructed {reconstructed_graph} out of {total_graph} graphs.")

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