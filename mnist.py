import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import random_split

import relu
import model_utils
import tester

EPSILON = 0.00000011920928955078125
torch.set_default_dtype(torch.float32)

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


#train_data, val_data = random_split(training_data, [0.8, 0.2])


# Create data loaders.
dataloaders = {}
dataloaders["train"] = DataLoader(training_data, batch_size=64)
dataloaders["val"] = DataLoader(test_data, batch_size=64)
dataloaders["test"] = DataLoader(test_data, batch_size=64)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
model_utils.replace_layers(model, [nn.ReLU], relu.ReLUCount(EPSILON))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

tester.run(dataloaders, model, optimizer, loss_fn, device, "test.csv", epochs=5, executions=3, relu_count=True)
print("Done!")