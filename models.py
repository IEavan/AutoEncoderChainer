"""
Module contains models to be used during training
"""

import torch
from torch.autograd import Variable

class ChainedAutoencoder(torch.nn.Module):
    """
    Class for chaining together multiple autoencoders
    such that the output of the first becomes the input of the second
    """
    def __init__(self):
        super(ChainedAutoencoder, self).__init__()

        # Activation function is leaky relu
        self.relu = torch.nn.functional.leaky_relu

        # Encoder structure will be fully connected
        # Dimensions are 3072 --> 1000 --> 100 --> 10 and vice versa for decoder
        dims = [3072, 1000, 100, 10, 100, 1000, 3072]
        self.linear_transforms = torch.nn.ModuleList([torch.nn.Linear(dims[i], dims[i+1])
                            for i in range(len(dims) - 1)])

    def forward(self, image_tensor, num_chains=1):
        # Store batch size for later reconstruction after flattening tensor
        batch_size = len(image_tensor)
        flat_tensor = image_tensor.view(batch_size, -1) # flattens tensor, with batch

        # Loop over the number of times to chain
        for _ in range(num_chains):
            for linear in self.linear_transforms:
                flat_tensor = linear(flat_tensor)
                flat_tensor = self.relu(flat_tensor)

        return flat_tensor.view(batch_size, 3, 32, 32)

def test():
    """ Test for class creation and forward method """
    model = ChainedAutoencoder()
    input_tensor = torch.FloatTensor(1, 3, 32, 32)
    model(Variable(input_tensor), 3)
    print("Test Success")

if __name__ == '__main__':
    test()
