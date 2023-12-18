
import torch.nn as nn
import torch.optim as optim



def train_ae(epoch, model, train_loader, device,conditional=False):
    """
    Entrena una epoca para el modelo Auto encoder
    """
    criterion = nn.MSELoss()
    model.train()
    train_loss = 0
    loss = None
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device=device)
        recon_batch = model(data)
        loss = criterion(recon_batch, data)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
    print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, loss.data))
    return(model)
