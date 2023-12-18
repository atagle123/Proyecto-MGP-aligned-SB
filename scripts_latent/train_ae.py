from latent_models.AE_model.trainning import train_ae
from latent_models.AE_model.model import AE


def train(train_loader,model,num_epochs=12,device="cpu"):

    for epoch in range(0, num_epochs):
        model=train_ae(epoch, model, train_loader,device=device)
    return(model)


def main(train_loader,num_epochs=12,input_size=28*28,hidden_dim=64,latent_size=15,device="cpu"):
    ae_model = AE(input_size, hidden_dim=hidden_dim, latent_size=latent_size)

    if device=="cuda":
        ae_model.cuda()

    ae_model=train(train_loader,ae_model,num_epochs,device)

    return(ae_model)


if __name__ == "__main__":
    main()
