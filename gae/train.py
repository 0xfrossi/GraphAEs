import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from dataset import StatiDatasetPadded
import time
from tqdm import tqdm
import numpy as np
from utils import count_parameters, gvae_loss_and_rec_accuracy
from gae import GVAE
from config import DEVICE as device

strings_1=[
        "{0} epoch {1} loss: {2:.6f} ", 
        "{0} epoch {1} ADJ MAT METRICS: acc: {2:.4f}, prec_mic: {3:.4f}, rec_mic: {4:.4f}",
        "{0} epoch {1} ATTR NODES METRICS: acc: {2:.4f}, prec_mic: {3:.4f}, rec_mic: {4:.4f}",
        "batch acc adj mat: {0:.5f}, batch acc att mat: {1:.5f}"
        ]

strings_2=[
        "reconstructed adj mat: {0}/{1} graphs ",
        "reconstructed attrs mat: {0}/{1} graphs ",
        "100% reconstructed graphs: {0}/{1} graphs ",
        ]

batch_dim=700
# Load data
print("Inizio caricamento dati...")
dataStati=StatiDatasetPadded(root="stati_data/data", filename="plans")
dataStati.shuffle()

train_data=dataStati[:800000] #920000
test_data= dataStati[800000:1000000]
final_test_data=dataStati[1000000:]

train_loader = DataLoader(train_data, batch_size=batch_dim)
test_loader = DataLoader(test_data, batch_size=batch_dim)
final_test_loader = DataLoader(final_test_data, batch_size=batch_dim)

print("Fine caricamento dati...")

# Load model
model = GVAE(feature_size=train_data[0].x.shape[1])
model = model.to(device)
print("Model parameters: ", count_parameters(model))

# Define loss and optimizer
loss_fn = gvae_loss_and_rec_accuracy            #0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-6)

# Train function
def run_one_epoch(data_loader, type, epoch,file_log, isFinal=False):
    # Store data
    all_batch_acc_adj=[]
    all_batch_acc_att=[]
    all_losses = []
    all_accs_attX = []
    all_accs_adj = []
    all_pre_attX=[]
    all_pre_adj=[]
    all_rec_attX=[]
    all_rec_adj=[]
    # Save some numbers
    total_graph = 0
    reconstructed_graph_att = 0
    reconstructed_graph_adj = 0
    reconstructed_graph=0

    # Iterate over data loader
    for i, batch in enumerate(tqdm(data_loader)):
                  
        batch=batch.to(device)  
        # Reset gradients
        optimizer.zero_grad() 
        # Call model
        logits_nodi, logits_adj = model(batch.x.float(),batch.edge_index,batch.batch) 
        
        # Calculate loss and backpropagate
        if type == "Train":
            loss,  acc_adj, acc_attx, num_recon_attx,  num_recon_adj,graph_rec_complete, score,acc_batch_adj, acc_batch_att = loss_fn(logits_nodi, logits_adj, batch.edge_index, batch.batch, batch.x.float(), batch.y)
            loss.backward()  
            optimizer.step() 
            all_losses.append(loss.detach().cpu().numpy())
        # Calculate metrics 
        elif type == "Test":
            loss,  acc_adj, acc_attx, num_recon_attx,  num_recon_adj,graph_rec_complete, score,acc_batch_adj, acc_batch_att = loss_fn(logits_nodi, logits_adj,batch.edge_index, batch.batch, batch.x.float(), batch.y )
            all_losses.append(loss.detach().cpu().numpy())

        total_graph = total_graph + batch.num_graphs
        reconstructed_graph_att= reconstructed_graph_att + num_recon_attx
        reconstructed_graph_adj= reconstructed_graph_adj + num_recon_adj
        reconstructed_graph=reconstructed_graph+graph_rec_complete
        # Store loss and metrics
        #all_losses.append(loss.detach().cpu().numpy())
        all_accs_attX.append(acc_attx)
        all_accs_adj.append(acc_adj)
        all_pre_attX.append(score[1])
        all_pre_adj.append(score[0])
        all_rec_attX.append(score[3])
        all_rec_adj.append(score[2])
        all_batch_acc_adj.append(acc_batch_adj.detach().cpu().numpy())
        all_batch_acc_att.append(acc_batch_att.detach().cpu().numpy())

    if isFinal==False:
        print(f"{type} epoch {epoch+1} loss: ", np.array(all_losses).mean())

    print(strings_1[1].format(type,epoch+1,np.array(all_accs_adj).mean(),np.array(all_pre_adj).mean(),np.array(all_rec_adj).mean()))
    print(strings_1[2].format(type,epoch+1,np.array(all_accs_attX).mean(), np.array(all_pre_attX).mean(),np.array(all_rec_attX).mean()))
    #print(strings_1[3].format(np.array(all_batch_acc_adj).mean(), np.array(all_batch_acc_att).mean()))
    
    
    print(strings_2[0].format(reconstructed_graph_adj,total_graph))
    print(strings_2[1].format(reconstructed_graph_att,total_graph))
    print(strings_2[2].format(reconstructed_graph,total_graph))
    
    #score=[pre_adj,pre_att,rec_adj,rec_att]
    path=''
    with open(file_log+'.txt', 'a') as f:
        
        f.write(strings_1[0].format(type,epoch+1,np.array(all_losses).mean()))
        f.write("\n")
        f.write(strings_1[1].format(type,epoch+1,np.array(all_accs_adj).mean(),np.array(all_pre_adj).mean(),np.array(all_rec_adj).mean()))
        f.write("\n")
        f.write(strings_1[2].format(type,epoch+1,np.array(all_accs_attX).mean(), np.array(all_pre_attX).mean(),np.array(all_rec_attX).mean()))
        #f.write(strings_1[3].format(np.array(all_batch_acc_adj).mean(), np.array(all_batch_acc_att).mean()))
        f.write("\n")
        f.write(strings_2[0].format(reconstructed_graph_adj,total_graph))
        f.write("\n")
        f.write(strings_2[1].format(reconstructed_graph_att,total_graph))
        f.write("\n")
        f.write(strings_2[2].format(reconstructed_graph,total_graph))
        f.write("\n\n")
    f.close()

    if type=="Train":
        return np.array(all_losses).mean()


# Run train
###########################

tolerance=6
n_epoch=50

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
        torch.save(model, f"{log_name}_best_model.pt")

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
        print("early-stopped\n")
        print("Final test epoch...")
        model=torch.load(f"{log_name}_best_model.pt")
        model=model.to(device)
        model.eval()
        run_one_epoch(final_test_loader, type="Test", epoch=epoch,file_log=log_name)
        break

    if epoch==n_epoch-1:
        print("Final test epoch...")
        model=torch.load(f"{log_name}_best_model.pt")
        model=model.to(device)
        model.eval()
        run_one_epoch(final_test_loader, type="Test", epoch=epoch,file_log=log_name)
