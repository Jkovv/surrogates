import os
import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

class DeepONet(nn.Module):
    def __init__(self, branch_dim, trunk_dim, latent_dim, hidden_size, act_branch, act_trunk):
        super().__init__()
        acts = {"ReLU": nn.ReLU(), "Tanh": nn.Tanh(), "SiLU": nn.SiLU(), "GELU": nn.GELU()}
        
        # branch net
        self.latent_dim = latent_dim
        self.branch = nn.Sequential(
            nn.Linear(branch_dim, hidden_size), acts[act_branch],
            nn.Linear(hidden_size, hidden_size), acts[act_branch],
            nn.Linear(hidden_size, latent_dim * 6) 
        )
        
        # trunk net
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, hidden_size), acts[act_trunk],
            nn.Linear(hidden_size, hidden_size), acts[act_trunk],
            nn.Linear(hidden_size, latent_dim)
        )
        
        # bias for each of the 6 cytokines
        self.bias = nn.Parameter(torch.zeros(6))

    def forward(self, x_b, x_t):
        batch_size = x_b.shape[0]
        seq_len = x_b.shape[1]       
        
        out_b = self.branch(x_b) # (Batch,Seq,latent_dim*6)
        out_b = out_b.view(batch_size, seq_len, self.latent_dim, 6) # (Batch,Seq,latent_dim,6)
        
        # trunk: (Grid_Points,latent_dim)
        out_t = self.trunk(x_t)
        
        # b - batch, s - sequence, l - latent, c - cytokine (6), g - grid points
        # result: (Batch,Sequence,Grid_Points,6)
        out = torch.einsum('bslc,gl->bsgc', out_b, out_t)
        
        return out + self.bias

def load_data_methodology(grid_size):
    path = f"preprocessed/{grid_size}x{grid_size}"

    data = np.load(os.path.join(path, "Y_target.npy")) 
    coords = np.load(os.path.join(path, "X_trunk.npy"))
    
    X_b, Y_t = [], []
    for t in range(data.shape[1] - 2):
        X_b.append(data[:, t:t+2, :, :, :]) #(N_sim,2,G,G,6)
        Y_t.append(data[:, t+2, :, :, :])   #(N_sim,G,G,6)
    
    # X_b: (N_sim,99,Input_Dim) 
    # Input_Dim = 2*G*G*6
    X_b = np.array(X_b).transpose(1, 0, 2, 3, 4, 5).reshape(data.shape[0], 99, -1)
    # Y_t: (N_sim,99,G*G,6)
    Y_t = np.array(Y_t).transpose(1, 0, 2, 3, 4).reshape(data.shape[0], 99, -1, 6)

    # 70/10/20 split
    n = X_b.shape[0]
    i_val, i_test = int(n * 0.7), int(n * 0.8)
    
    return (X_b[:i_val], Y_t[:i_val]), \
           (X_b[i_val:i_test], Y_t[i_val:i_test]), \
           (X_b[i_test:], Y_t[i_test:]), coords
