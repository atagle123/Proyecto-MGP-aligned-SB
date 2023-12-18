import os

import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
import pathlib

import sbalign.utils.helper as helper
from sbalign.utils.sb_utils import sample_from_brownian_bridge
from sbalign.training.diffusivity import get_diffusivity_schedule


def build_data_loader(args):
    if args.transform is None:
        transform = BrownianBridgeTransform(g=get_diffusivity_schedule(args.diffusivity_schedule, args.max_diffusivity))

    if helper.is_custom_dataset(args.dataset):
        train_dataset = SavedDataset(args.dataset, transform=transform, n_samples=args.n_samples, mode="train", root_dir="../reproducibility/")
        val_dataset = SavedDataset(args.dataset, transform=transform, n_samples=args.n_samples, mode="val", root_dir="../reproducibility/")


    else:
        raise ValueError(f"{args.dataset} is not supported")

    train_loader = train_dataset.create_loader(batch_size=args.train_bs, 
                                               num_workers=args.num_workers, 
                                               shuffle=True)
    val_loader = val_dataset.create_loader(batch_size=args.val_bs,
                                           num_workers=args.num_workers,
                                           shuffle=True)
    return train_loader, val_loader


class BrownianBridgeTransform(BaseTransform):

    def __init__(self, g):
        self.g = g

    def __call__(self, data):
        bs = data.pos_0.shape[0]
        t = torch.rand((bs, 1))
        return self.apply_transform(data, t)

    def apply_transform(self, data, t):
        # assert (data.pos_0[:,1] == data.pos_T[:,1]).all(), (data.pos_0[:,1], data.pos_T[:,1])
        data.pos_t = sample_from_brownian_bridge(g=self.g, t=t, x_0=data.pos_0, x_T=data.pos_T, t_min=0.0, t_max=1.0)
        data.t = t
        return data


class SavedDataset(Dataset):

    def __init__(self, dataset_name, transform=None, n_samples: int=10000, mode="train", root_dir=pathlib.Path(__file__).parent.resolve(), device='cpu'):
        self.dataset_name = dataset_name
        self.transform = transform
        self.mode = mode

        self.n_samples = n_samples
        self.mode = mode

        dataset_root_path = os.path.join(root_dir, dataset_name)

        self.data = {
            time: np.load(os.path.join(dataset_root_path, "data", self.get_partition_name(time)))[:n_samples] for time in ['initial', 'final']
        }

        # Convert into torch tensors
        self.data = {
            time: torch.from_numpy(self.data[time]).to(device) for time in self.data
        }

    def get_partition_name(self, time):
        return f"{self.dataset_name}_embs_{time}_{self.mode}.npy"

    def __len__(self):
        return self.data["initial"].shape[0]

    def __getitem__(self, index):
        return (self.data["initial"][index], self.data["final"][index])

    def collate_fn(self, data_batch):
        data = Data()
        pos_0, pos_T = zip(*data_batch)

        data.pos_0 = torch.stack(pos_0, dim=0)
        data.pos_T = torch.stack(pos_T, dim=0)
        assert data.pos_0.shape == data.pos_T.shape

        if self.transform is not None:
            return self.transform(data)
        return data

    def create_loader(self,
                      batch_size: int,
                      num_workers: int,
                      shuffle: bool = False):
        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn)

        

class CheckerBoard:

    def __init__(self, n_samples):
        self.n_samples = n_samples

    def sample(self):
        n = self.n_samples
        n_points = 3 * n
        n_classes = 2
        freq = 5
        x = np.random.uniform(
            -(freq // 2) * np.pi, (freq // 2) * np.pi, size=(n_points, n_classes)
        )
        mask = np.logical_or(
            np.logical_and(np.sin(x[:, 0]) > 0.0, np.sin(x[:, 1]) > 0.0),
            np.logical_and(np.sin(x[:, 0]) < 0.0, np.sin(x[:, 1]) < 0.0),
        )
        y = np.eye(n_classes)[1 * mask]
        x0 = x[:, 0] * y[:, 0]
        x1 = x[:, 1] * y[:, 0]
        sample = np.concatenate([x0[..., None], x1[..., None]], axis=-1)
        sqr = np.sum(np.square(sample), axis=-1)
        idxs = np.where(sqr == 0)
        samples = np.delete(sample, idxs, axis=0)
        # res=res+np.random.randn(*res.shape)*1
        samples = samples[0:n, :]

        # transform dataset by adding constant shift
        samples_t = samples + np.array([0, 3])

        return {"initial": torch.Tensor(samples_t).float(),
                "final": torch.Tensor(samples).float()}






class MatchingWithException:

    def __init__(self, n_samples):
        self.n_samples = n_samples

    def sample(self):
        n = self.n_samples

        left_cloud = np.stack([np.random.normal(-5, .25, size=(n,)), np.linspace(-3.5, 3.5, n)], axis=1)
        right_cloud = np.stack([np.random.normal(5, .25, size=(n,)), np.linspace(-3.5, 3.5, n)], axis=1)

        exceptions_num = np.maximum((25*n)//100, 1)
        print(exceptions_num)

        left_cloud = np.roll(left_cloud, exceptions_num, axis=0)

        # TODO: Check that SyntheticDataset shuffles all samples before extracting train/val slices 
        rand_shuffling = np.random.permutation(left_cloud.shape[0])
        left_cloud = left_cloud[rand_shuffling]
        right_cloud = right_cloud[rand_shuffling]

        return {
            "initial": torch.from_numpy(left_cloud).float(),
            "final": torch.from_numpy(right_cloud).float()
        }




class DataSampler:  # a dump data sampler

    def __init__(self, dataset, n_samples, device):
        self.num_sample = len(dataset)
        self.dataloader = setup_loader(dataset, n_samples)
        self.n_samples = n_samples
        self.device = device

    def sample(self):
        data = next(self.dataloader)
        return data.to(self.device)


def setup_loader(dataset, n_samples):
    train_loader = DataLoaderX(
        dataset, n_samples=n_samples, shuffle=True, num_workers=0, drop_last=True
    )
    print("number of samples: {}".format(len(dataset)))

    while True:
        yield from train_loader




def rotate2d(x, radians):
    """Build a rotation matrix in 2D, take the dot product, and rotate."""
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, x)

    return m
