import torch
from torch.autograd import Variable

class ChainedAutoencoder(torch.nn.Module):
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
        batch_size = len(image_tensor)
        flat_tensor = image_tensor.view(batch_size, -1) # flattens tensor, with batch

        for chain_id in range(num_chains):
            for linear in self.linear_transforms:
                flat_tensor = linear(flat_tensor)
                flat_tensor = self.relu(flat_tensor)

        return flat_tensor.view(batch_size,3,32,32)

if __name__ == '__main__':
    model = ChainedAutoencoder()
    x = torch.FloatTensor(1,3,32,32)
    y = model(Variable(x), 1)
