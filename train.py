import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import ChainedAutoencoder

USE_CUDA = torch.cuda.is_available()
TRAIN_LOADER = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                        transform=transforms.ToTensor()))
PRINT_EVERY = 500

def setup(learning_rate=1e-3):
    model = ChainedAutoencoder()
    if USE_CUDA:
        model = model.cuda()

    # Option? Add weight loading

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    return model, optimizer, criterion

def get_optimizer(model, learning_rate=1e-3):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_example(model, image_tensor, optimizer, criterion, num_chains=1):
    image_variable = Variable(image_tensor)
    if USE_CUDA:
        image_variable = image_variable.cuda()
    image_output = model(image_variable, num_chains=num_chains)
    loss = criterion(image_variable, image_output)
    loss.backwards()
    optimizer.step()
    optimizer.zero_grad()
    return loss

if __name__ == '__main__':
    unchained_model, unchained_optimizer, criterion = setup()
    unchained_losses = []

    chained_model, chained_optimizer, _ = setup()
    chained_losses = []

    for iter_id, (image_tensor, label) in enumerate(TRAIN_LOADER):
        unchained_losses.append(train_example(
            unchained_model, image_tensor, unchained_optimizer, criterion))

        num_chains = (iter_id // 10000) + 1
        chained_losses.append(train_example(
            chained_model, image_tensor, chained_optimizer,
            criterion, num_chains=num_chains))


        if (iter_id + 1) % PRINT_EVERY == 0:
            print("Current unchained loss is {}".format(
                sum(unchained_losses[-PRINT_EVERY:]) / PRINT_EVERY))
            print("Current average {}-chained loss is {}".format(
                num_chains, sum(chained_losses[-PRINT_EVERY:]) / PRINT_EVERY))
