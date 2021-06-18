import os
import torch
from torchvision import datasets, transforms

def get_mnist(batch_size=32, target_directory="./data/") -> ("train_loader, valid_loader"):
    """
    Return the mnist dataset
    """

    if not os.path.exists(target_directory):
        os.mkdir(target_directory)

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            target_directory, train=True,
            download=True, transform=transforms.ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            target_directory, train=False,
            transform=transforms.ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, valid_loader
