from latent_models.VAE_model.trainning import train_vae
from latent_models.VAE_model.model import VAE


def train(train_loader,model,num_epochs=12,device="cpu"):

    for epoch in range(0, num_epochs):
        model=train_vae(epoch, model, train_loader,device=device)
    return(model)


def main(train_loader,num_epochs=12,input_size=28*28,hidden_dim=64,latent_size=15,device="cpu"):
    vae_model = VAE(input_size, hidden_dim=hidden_dim, latent_size=latent_size)

    if device=="cuda":
        vae_model.cuda()

    vae_model=train(train_loader,vae_model,num_epochs,device)

    return(vae_model)


if __name__ == "__main__":
    main()
