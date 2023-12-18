import torch
import torch.nn as nn

def reparameterization_trick(mu, logvar):
    """
        Inputs:
            - mu: Tensor con medias, (N, Z)
            - logvar: Tensor con las log-varianza, (N, Z)

        Retorna:
            - z: muestra del vector latente
    """
    # --------------------------------------------------------------------------
    std = torch.exp(0.5 * logvar)  #calcular desviacion estandar
    e_normal=torch.randn_like(std) # samplear normal (0,1)
    z = mu+e_normal*std  # reparametrizar la normal
    # --------------------------------------------------------------------------
    return z

class VAE(nn.Module):
    """
        Fully Connected VAE.
    """
    def __init__(self, input_size, hidden_dim=64, latent_size=16):
        super(VAE, self).__init__()
        self.input_size = input_size    # Height * Width
        self.latent_size = latent_size  # z
        self.hidden_dim = hidden_dim    # H

        # Bloque Encoder
        # ----------------------------------------------------------------------
        self.encoder=nn.Sequential(
            nn.Linear(self.input_size, self.hidden_dim),   # bloque encoder de capas lineales y relu
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        #    logvar_layer

        self.mu_layer = nn.Linear(self.hidden_dim, self.latent_size)   # calculo de capa de mu y logvar
        self.logvar_layer = nn.Linear(self.hidden_dim, self.latent_size)
        # ----------------------------------------------------------------------


        # Bloque Decoder
        # ----------------------------------------------------------------------
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_dim),  #bloque decoder de capas lineales, relu y sigmoid alfinal
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_size),
            nn.Sigmoid(),
            nn.Unflatten(1,(1,28,28))    # unflatten para ver la imagen
        )
        # ----------------------------------------------------------------------


    def forward(self, x):
        # ----------------------------------------------------------------------
        #  Implemente el forward pass considerando los siguientes pasos:
        encode=self.encoder(x.view(-1, 784))               # encode el input
        mu,logvar=self.mu_layer(encode),self.logvar_layer(encode) # obtener mu y logvar
        z=reparameterization_trick(mu,logvar)    # samplar de z
        x_hat=self.decoder(z)     # reconstruir la imagen
        # ----------------------------------------------------------------------
        return x_hat, mu, logvar
