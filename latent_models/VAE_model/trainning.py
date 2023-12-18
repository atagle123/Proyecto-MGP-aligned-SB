import torch
import torch.nn.functional as F
import torch.optim as optim


def loss_function(x_hat, x, mu, logvar):
    """
    Computa la cota inferior de la evidencia negativa como terminos de la función
    de costo del VAE.

    Inputs:
    - x_hat: Data reconstruida del input con shape (N, 1, H, W)
    - x: Data input a evaluar con shape (N, 1, H, W)
    - mu: Matriz con las estimaciones de mu posterior (N, Z), con Z como
      dimensión del espacio latente
    - logvar: Matriz con las estimaciones de la varianza en espacio logaritmico (N, Z),
      con Z como dimensión del espacio latente

    Retorna:
    - loss: Tensor que contiene el escalar que representa la pérdida de la cota
      inferior de evidencia negativa
    """
    # --------------------------------------------------------------------------
    reconstruction_term = F.binary_cross_entropy(x_hat.view(-1, 784),x.view(-1, 784), reduction='sum')/x.shape[0]       # binary cross entropy
    kl_divergence_term=torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1))    # divergencia kl para normal (0,1) y normal con media y desv estandar arbitrarias
    # --------------------------------------------------------------------------
    loss = reconstruction_term + kl_divergence_term
    return loss



def train_vae(epoch, model, train_loader, device,conditional=False):
    """
    Entrena una epoca para el modelo VAE o ConditionalVAE

    Inputs:
    - epoch: # de epoca
    - model: VAE o ConditionalVae PyTorch model
    - train_loader: PyTorch Dataloader con la data de entrenamiento
    - conditional: Boolean que representa True para ConditionalVae, sino VAE
    """
    model.train()
    train_loss = 0
    num_classes = 10
    loss = None
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device=device)
        if conditional:
            one_hot_vec = torch.eye(num_classes)[labels].to(device=device)
            recon_batch, mu, logvar = model(data, one_hot_vec)
        else:
            recon_batch, mu, logvar = model(data)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
    print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, loss.data))
    return(model)