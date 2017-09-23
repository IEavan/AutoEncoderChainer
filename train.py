import torch
from torchvision import datasets, transforms

train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                        transform=transforms.ToTensor()))

for data, target in train_loader:
    print("data is {}".format(data))
    print("label is {}".format(target))
    break
