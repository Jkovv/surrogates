import torch
import torch.nn as nn
import torch.optim as optim
from core import DeepONet, DEVICE, set_seed

def train_and_eval(params, train_set, val_set, coords, seed):
    set_seed(seed)
    X_train, Y_train = train_set
    X_val, Y_val = val_set
    
    model = DeepONet(
        X_train.shape[2], 
        coords.shape[1], 
        params['latent_dim'], 
        params['hidden_size'], 
        params['act_branch'], 
        params['act_trunk']
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.MSELoss()
    
    X_t = torch.tensor(X_train).to(DEVICE)
    Y_t = torch.tensor(Y_train).to(DEVICE)
    X_v = torch.tensor(X_val).to(DEVICE)
    Y_v = torch.tensor(Y_val).to(DEVICE)
    c_t = torch.tensor(coords).to(DEVICE)
    
    # early stopping
    best_val_loss = float('inf')
    best_state = None
    patience = 12 
    counter = 0

    for epoch in range(80):
        model.train()
        optimizer.zero_grad()
        pred = model(X_t, c_t)
        loss = criterion(pred, Y_t)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.6f}")
        
        # val loss
        model.eval()
        with torch.no_grad():
            v_pred = model(X_v, c_t)
            current_val_loss = criterion(v_pred, Y_v).item()
        
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()} # best state
            counter = 0
        else:
            counter += 1
            
        if counter >= patience:
            break
        
    return best_val_loss, best_state