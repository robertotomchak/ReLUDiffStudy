import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import random_split

import relu
import model_utils
import tester

EPSILON = 0.00000011920928955078125
torch.set_default_dtype(torch.float32)

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        ToTensor(),
    ])
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        ToTensor(),
    ])
)

INPUT_SHAPE = (3, 28, 28)
NUM_CLASSES = 10


train_data, val_data = random_split(training_data, [0.7, 0.3])


# Create data loaders.
dataloaders = {}
dataloaders["train"] = DataLoader(train_data, batch_size=32, shuffle=True)
dataloaders["val"] = DataLoader(val_data, batch_size=1024, shuffle=False)
dataloaders["test"] = DataLoader(test_data, batch_size=1024, shuffle=False)

# Get cpu, gpu or mps device for training.
device = (
    "cuda:1"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
model = model_utils.Hochuli(INPUT_SHAPE, NUM_CLASSES, relu.ReLU6Count(EPSILON))

model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9)

tester.run(dataloaders, model, optimizer, loss_fn, device, "hochuli/mnist_count.csv", epochs=30, executions=5, relu_count=True)
print("Done!")

