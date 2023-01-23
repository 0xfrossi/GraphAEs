import torch
from torch_geometric.loader import DataLoader
from dataset import DepotsDataset
import time
#from Codice.logistic_data.dataset import LogisticDataset

from tqdm import tqdm
import numpy as np
from utils import count_parameters, reconstruction_accuracy, slice_edge_type_from_edge_feats, gvae_loss_and_rec_accuracy
from gvae import GVAE
from config import DEVICE as device

strings_1=[
        "{0} epoch {1} loss: {2:.5f} \n", 
        "{0} epoch {1} accuracy adj mat. : {2:.5f} \n",
        "{0} epoch {1} accuracy mat. type 'at1': {2:.5f} \n",
        "{0} epoch {1} accuracy mat. type 'at2': {2:.5f} \n",
        "{0} epoch {1} accuracy mat. type 'empty1': {2:.5f} \n"]
strings_2=[
        " Adj mat. Reconstructed {0} out of {1} graphs.\n",
        " Type 'at1' Reconstructed {0} out of {1} graphs.\n",
        " Type 'at2' Reconstructed {0} out of {1} graphs.\n",
        " Type 'empty1' Reconstructed {0} out of {1} graphs.\n",
        "Graphs with all properties reconstructed at 100%: {0} out of {1} graphs.\n",
        "**************************  END EPOCH  **************************************\n"]



#200
batch_dim=1
# Load data
print("Inizio caricamento dati...")
train_data= DepotsDataset(root="depots_data/data/", filename="depots_plans")[:140000] #0.7
test_data= DepotsDataset(root="depots_data/data/", filename="depots_plans")[140000:150000]
final_test_data=DepotsDataset(root="depots_data/data/", filename="depots_plans")[150000:240000]

train_data.shuffle()
test_data.shuffle()

train_loader = DataLoader(train_data, batch_size=batch_dim, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_dim, shuffle=True)
final_test_loader = DataLoader(final_test_data, batch_size=batch_dim, shuffle=True)

print("Fine caricamento dati...")

# Load model
model = GVAE(feature_size=train_data[0].x.shape[1])
model = model.to(device)
print("Model parameters: ", count_parameters(model))

# Define loss and optimizer
loss_fn = gvae_loss_and_rec_accuracy            #0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-6)

# Train function
def run_one_epoch(data_loader, type, epoch,file_log):
    # Store per batch loss and accuracy 
    
    all_losses = []
    all_accs_0 = []
    all_accs_1 = []
    all_accs_2 = []
    all_accs_adj = []
    # Save some numbers
    total_graph = 0
    reconstructed_graph_0 = 0
    reconstructed_graph_1 = 0
    reconstructed_graph_2 = 0
    reconstructed_graph_adj = 0
    reconstructed_graph=0

    # Iterate over data loader
    for i, batch in enumerate(tqdm(data_loader)):
       
           
        batch=batch.to(device)  
        # Reset gradients
        optimizer.zero_grad() 
        # Call model
        mat_logits_0, mat_logits_1, mat_logits_2 = model(batch.x.float(), batch.edge_attr.float(),batch.edge_index,batch.batch) 
        discrete_edge_attr=slice_edge_type_from_edge_feats(batch.edge_attr.float())
        
        # Calculate loss and backpropagate
        #loss,  acc_adj,acc_0,acc_1, num_recon_0, num_recon_1,num_recon_adj = loss_fn(mat_logits_adj,mat_logits_0,mat_logits_1, batch.edge_index, batch.batch, discrete_edge_attr )
        if type == "Train":
            loss,  acc_adj,acc_0,acc_1, acc_2, num_recon_0, num_recon_1,num_recon_2, num_recon_adj,graph_rec_complete = loss_fn(mat_logits_0,mat_logits_1,mat_logits_2, batch.edge_index, batch.batch, discrete_edge_attr )
            loss.backward()  
            optimizer.step() 
            all_losses.append(loss.detach().cpu().numpy())
        # Calculate metrics
        else:
            acc_adj, acc_0, acc_1, acc_2, num_recon_0, num_recon_1,num_recon_2,num_recon_adj,graph_rec_complete = reconstruction_accuracy(mat_logits_0, mat_logits_1,mat_logits_2,batch.edge_index, batch.batch,discrete_edge_attr)
        
        total_graph = total_graph + batch.num_graphs

        reconstructed_graph_0 = reconstructed_graph_0 + num_recon_0
        reconstructed_graph_1 = reconstructed_graph_1 + num_recon_1
        reconstructed_graph_2 = reconstructed_graph_2 + num_recon_2
        reconstructed_graph_adj= reconstructed_graph_adj + num_recon_adj
        reconstructed_graph=reconstructed_graph+graph_rec_complete
        # Store loss and metrics
        #all_losses.append(loss.detach().cpu().numpy())
        all_accs_0.append(acc_0.detach().cpu().numpy())
        all_accs_1.append(acc_1.detach().cpu().numpy())
        all_accs_2.append(acc_2.detach().cpu().numpy())
        all_accs_adj.append(acc_adj.detach().cpu().numpy())

    if type=="Train":
        print(f"{type} epoch {epoch+1} loss: ", np.array(all_losses).mean())

  
    print(f"{type} epoch {epoch+1} accuracy adj mat. : ", np.array(all_accs_adj).mean())
    print(f"{type} epoch {epoch+1} accuracy mat. type 'at1': ", np.array(all_accs_0).mean())
    print(f"{type} epoch {epoch+1} accuracy mat. type 'at2': ", np.array(all_accs_1).mean())
    print(f"{type} epoch {epoch+1} accuracy mat. type 'empty1': ", np.array(all_accs_2).mean())

    print(f" Adj mat. Reconstructed {reconstructed_graph_adj} out of {total_graph} graphs.")
    print(f" Type 'at1' Reconstructed {reconstructed_graph_0} out of {total_graph} graphs.")
    print(f" Type 'at2' Reconstructed {reconstructed_graph_1} out of {total_graph} graphs.")
    print(f" Type 'empty1' Reconstructed {reconstructed_graph_2} out of {total_graph} graphs.")
    print(f"Graphs with all properties reconstructed at 100%: {reconstructed_graph} out of {total_graph} graphs.")
    print("\n***************************  END EPOCH **************************************\n")

    path='/home/frossi/Codice/depots_AE/'
    with open(path+file_log+'.txt', 'a') as f:
        if type=="Train":
            f.write(strings_1[0].format(type,epoch+1,np.array(all_losses).mean()))

        f.write(strings_1[1].format(type,epoch+1,np.array(all_accs_adj).mean()))
        f.write(strings_1[2].format(type,epoch+1,np.array(all_accs_0).mean()))
        f.write(strings_1[3].format(type,epoch+1,np.array(all_accs_1).mean()))
        f.write(strings_1[4].format(type,epoch+1,np.array(all_accs_2).mean()))

        f.write(strings_2[0].format(reconstructed_graph_adj,total_graph))
        f.write(strings_2[1].format(reconstructed_graph_0,total_graph))
        f.write(strings_2[2].format(reconstructed_graph_1,total_graph))
        f.write(strings_2[3].format(reconstructed_graph_2,total_graph))
        f.write(strings_2[4].format(reconstructed_graph,total_graph))
        f.write(strings_2[5])
        f.write("\n")
    f.close()

    if type=="Train":
        return np.array(all_losses).mean()

# Run training
###########################

tolerance=6
n_epoch=200

###########################
log_name = time.strftime("%Y%m%d-%H%M%S")
best_loss=np.inf
prev_loss= np.inf
count=0
stop=False
for epoch in range(n_epoch): 
    model.train()
    print("Start train epoch...")
    this_loss=run_one_epoch(train_loader, type="Train", epoch=epoch,file_log=log_name)

    if this_loss < best_loss:
        best_loss= this_loss
        count=0
        torch.save(model.state_dict(), "/home/frossi/Codice/depots_AE/saved_states/"+"best_depots.pt")

    if this_loss> best_loss:
        count+=1 
        if count >= tolerance: 
            stop=True
             
    prev_loss=this_loss

    if epoch % 5 == 0 and epoch!=0:
        print("Start test epoch...")
        model.eval()
        run_one_epoch(test_loader, type="Test", epoch=epoch,file_log=log_name)
    
    if stop== True:
        print("early-stop\n")
        print("Final test epoch...")
        model.load_state_dict(torch.load("/home/frossi/Codice/depots_AE/saved_states/best_depots.pt"))
        model.eval()
        run_one_epoch(final_test_loader, type="Test", epoch=epoch,file_log=log_name)
        break

    if epoch==n_epoch-1:
        print("Final test epoch...")
        model.load_state_dict(torch.load("/home/frossi/Codice/depots_AE/saved_states/best_depots.pt"))
        model.eval()
        run_one_epoch(final_test_loader, type="Test", epoch=epoch,file_log=log_name)
