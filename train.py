import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import ChainedAutoencoder

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
CHAIN_LENGTH_INCREMENT_TIME = 5000

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

def lr_schedule(initial_lr=1e-3, decay_rate=0.5):
    lr = initial_lr / decay_rate
    while True:
        yield lr * decay_rate

def train_example(model, image_tensor, criterion, optimizer=None,
        num_chains=1, backprop=True):
    
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
    print("Evaluating Models")
    chained_losses = []
    unchained_losses = []
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
    leanring_rate_schedule = lr_schedule()
    learning_rate = leanring_rate_schedule.__next__()

    unchained_model, unchained_optimizer, criterion = setup(learning_rate=learning_rate)
    unchained_losses = []

    chained_model, chained_optimizer, _ = setup(learning_rate=learning_rate)
    chained_losses = []

    print("Starting Training")
    for iter_id, (image_tensor, label) in enumerate(TRAIN_LOADER):
        unchained_losses.append(train_example(
            unchained_model, image_tensor, criterion, 
            optimizer=unchained_optimizer))

        num_chains = (iter_id // CHAIN_LENGTH_INCREMENT_TIME) + 1
        chained_losses.append(train_example(
            chained_model, image_tensor, criterion,
            optimizer=chained_optimizer, num_chains=num_chains))
        
        if (iter_id + 1) % LEANRING_RATE_DECAY_TIME == 0:
            new_learning_rate = leanring_rate_schedule.__next__()
            unchained_optimizer = get_optimizer(model=unchained_model,
                    learning_rate=new_learning_rate)
            chained_optimizer = get_optimizer(model=chained_model,
                    learning_rate=new_learning_rate)
            print("Learning rate updated to {}\n".format(new_learning_rate))

        if (iter_id + 1) % PRINT_EVERY == 0:
            print("Current average unchained loss is {:.3}".format(
                sum(unchained_losses[-PRINT_EVERY:]) / PRINT_EVERY))
            print("Current average {}-chained loss is {:.3}".format(
                num_chains, sum(chained_losses[-PRINT_EVERY:]) / PRINT_EVERY))

    evaluate(unchained_model, chained_model, criterion)
