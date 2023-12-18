from torchvision import  transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
from torch.utils.data import DataLoader, ConcatDataset 




def obtain_mnist_dataset(rotation=False,min_angle=90,max_angle=90):

    if rotation:

        rotate_transform = transforms.Compose([
            transforms.RandomRotation((min_angle, max_angle)),
            transforms.ToTensor()
        ])

        train_dataset = dataset.MNIST('data', train=True, download=True,
                                        transform=rotate_transform)

    else:
        train_dataset = dataset.MNIST('data', train=True, download=True,
                                       transform=transforms.ToTensor())

    return train_dataset



def concat_datasets(set1,set2,batch_size=128,shuffle=True):
    merged_dataset = ConcatDataset([set1, set2])

    train_loader = DataLoader(merged_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=2)

    return train_loader
