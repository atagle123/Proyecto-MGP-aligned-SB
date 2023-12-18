import torch.nn as nn





class AE(nn.Module):
    """
        Fully Connected AE.
    """
    def __init__(self, input_size, hidden_dim=64, latent_size=16):
        super(AE, self).__init__()
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
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_size),
            nn.ReLU()
        )

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
        x_hat=self.decoder(encode)     # reconstruir la imagen
        # ----------------------------------------------------------------------
        return x_hat
