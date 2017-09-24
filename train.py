"""
module for training the ChainedAutoencoder model from models.py
and evaluate the loss compared to a non chained autoencoder
"""

import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import ChainedAutoencoder

# Constants and Data
USE_CUDA = torch.cuda.is_available()
TRAIN_LOADER = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                        transform=transforms.ToTensor()),
        shuffle=True)
TEST_LOADER = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False,
            transform=transforms.ToTensor()),
        shuffle=True)
PRINT_EVERY = 50
LEANRING_RATE_DECAY_TIME = 2000
DECAY = 0.8
CHAIN_LENGTH_INCREMENT_TIME = 5000

def setup(learning_rate=1e-3):
    """ returns necessary objects for training a single model
    return: model, optimizer, criterion
    """
    model = ChainedAutoencoder()
    if USE_CUDA:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    return model, optimizer, criterion

def get_optimizer(model, learning_rate=1e-3):
    """ returns an optimizer for the given model
    configured with the given learning_rate """
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

def lr_schedule(initial_lr=1e-3, decay_rate=DECAY):
    """ returns generator which yields exponentially decreasing learning_rates """
    lr = initial_lr / decay_rate
    while True:
        lr *= decay_rate
        yield lr

def train_example(model, image_tensor, criterion, optimizer=None,
        num_chains=1, backprop=True):
    """ Trains the model on a single image tensor and returns the loss """

    # Require optimizer if backprobagating gradients
    if backprop:
        assert optimizer is not None

    image_variable = Variable(image_tensor, requires_grad=False)
    if USE_CUDA:
        image_variable = image_variable.cuda()

    image_output = model(image_variable, num_chains=num_chains)
    loss = criterion(image_output, image_variable)

    if backprop:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss.data[0]

def evaluate(unchained_model, chained_model, criterion):
    """ Evaluate the models on the test set and print average loss """
    
    print("Evaluating Models")
    chained_losses = []
    unchained_losses = []
    
    # Loop over test set and compute loss without backpropagting
    for iter_id, (image_tensor, label) in enumerate(TEST_LOADER):
        chained_losses.append(train_example(
            chained_model, image_tensor, criterion, backprop=False))
        unchained_losses.append(train_example(
            unchained_model, image_tensor, criterion, backprop=False))

    print("Unchained AutoEncoder achieved {:.3} test set loss".format(
        sum(unchained_losses) / len(TEST_LOADER)))
    print("Chained AutoEncoder achieved {:.3} test set loss".format(
        sum(chained_losses) / len(TEST_LOADER)))

if __name__ == '__main__':

    # Setup learning rate scheduler and the two models
    leanring_rate_schedule = lr_schedule()
    learning_rate = leanring_rate_schedule.__next__()

    unchained_model, unchained_optimizer, criterion = setup(learning_rate=learning_rate)
    unchained_losses = []

    chained_model, chained_optimizer, _ = setup(learning_rate=learning_rate)
    chained_losses = []

    print("Starting Training")

    # Loop over training data
    for iter_id, (image_tensor, label) in enumerate(TRAIN_LOADER):

        # train on single image_tensor
        unchained_losses.append(train_example(
            unchained_model, image_tensor, criterion, 
            optimizer=unchained_optimizer))

        # determine number of chains to use and train on image_tensor
        num_chains = (iter_id // CHAIN_LENGTH_INCREMENT_TIME) + 1
        chained_losses.append(train_example(
            chained_model, image_tensor, criterion,
            optimizer=chained_optimizer, num_chains=num_chains))

        # update optimizers with new learning rate
        if (iter_id + 1) % LEANRING_RATE_DECAY_TIME == 0:
            new_learning_rate = leanring_rate_schedule.__next__()
            unchained_optimizer = get_optimizer(model=unchained_model,
                    learning_rate=new_learning_rate)
            chained_optimizer = get_optimizer(model=chained_model,
                    learning_rate=new_learning_rate)
            print("Learning rate updated to {}\n".format(new_learning_rate))

        # print training progress
        if (iter_id + 1) % PRINT_EVERY == 0:
            print("Current average unchained loss is {:.3}".format(
                sum(unchained_losses[-PRINT_EVERY:]) / PRINT_EVERY))
            print("Current average {}-chained loss is {:.3}".format(
                num_chains, sum(chained_losses[-PRINT_EVERY:]) / PRINT_EVERY))

    # evaluate the accuracy of the two models on the test data
    evaluate(unchained_model, chained_model, criterion)
