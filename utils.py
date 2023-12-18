import os
import math
import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['font.size'] = 16
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def save_data(data,train_prop=0.7,val_prop=0.15,time="initial",folder="custom_dataset"): ## falta agregar directorio y nombre de carpeta 
    """
    Guarda los datos en archivos .npy con nombres específicos.
    Los conjuntos de entrenamiento, validación y prueba se dividen en proporción proporcionada.

    data: Arreglo de numpy con los datos a guardar. (N_data,dims)
    train_prop: Proporción de los datos para el conjunto de entrenamiento.
    val_prop: Proporción de los datos para el conjunto de validación.
    time: String con la hora en que se guardan los datos.
    
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the folder if it doesn't exist
    folder_path = os.path.join(current_dir,"reproducibility", folder,"data")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    total_samples = data.shape[0]
    train_samples = int( train_prop* total_samples)  # 70% para entrenamiento
    val_samples = int(val_prop * total_samples)   # 15% para validación
    test_samples = total_samples - train_samples - val_samples  # Lo restante para prueba

    # Divide el arreglo en conjuntos de entrenamiento, validación y prueba
    train_data = data[:train_samples]
    val_data = data[train_samples:train_samples + val_samples]
    test_data = data[train_samples + val_samples:]
 
    train_name="".join([folder,"_embs_", time, "_train.npy"])
    val_name="".join([folder,"_embs_", time, "_val.npy"])
    test_name="".join([folder,"_embs_", time, "_test.npy"])
        
    # Guarda los conjuntos de datos en archivos .npy
    np.save(os.path.join(folder_path, train_name), train_data)
    np.save(os.path.join(folder_path, val_name), val_data)
    np.save(os.path.join(folder_path, test_name), test_data)

def show_batch(images, CMAP='gray'):
    images = images.view(images.shape[0], -1)
    sqrtn = int(math.ceil(math.sqrt(images.shape[0])))
    sqrt_img = int(math.ceil(math.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.imshow(img.reshape([sqrt_img, sqrt_img]), cmap=CMAP)
    return

def latent_data_to_npy(model,data_loader,device,input_dim=28*28,model_type="VAE"):
    """
    model tiene que tener bloque encoder
    """
    model.eval()
    all_latent_vars = []
    with torch.no_grad():
        if model_type == "VAE":
            for data,labels in data_loader:
                data = data.to(device=device)
                recon_batch, mu, logvar = model(data)
                all_latent_vars.append(torch.cat((mu, logvar), dim=1).cpu().detach().numpy())

        else:
            for data,labels in data_loader:
                data = data.to(device=device)
                recon_batch = model.encoder(data.view(-1, input_dim))
                all_latent_vars.append(recon_batch.cpu().detach().numpy())

    all_latent_vars = np.concatenate(all_latent_vars, axis=0)
    return (all_latent_vars)


def save_model(model,model_name="model_1",folder="latent_models\saved_models"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_name="".join([model_name, ".pt"])
    dir = os.path.join(current_dir,folder,model_name)
    torch.save(model.state_dict(), dir) 


def load_model(model,model_name="model_1",folder="latent_models\saved_models"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_name="".join([model_name, ".pt"])
    dir = os.path.join(current_dir,folder,model_name)
    model.load_state_dict(torch.load(dir))
    return model


def get_data_bridge_inference(folder="custom_dataset",stage="test"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir,"reproducibility", folder,"data")

    name_data="".join([folder,"_embs_initial_", stage, ".npy"])
    os.path.join(folder_path, name_data)

    data_initial=np.load(os.path.join(folder_path, name_data))
    data_initial=torch.from_numpy(data_initial)
    return(data_initial)