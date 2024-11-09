import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from random import randint

import relu
import model_utils
import tester

EXECUTIONS = 5
EPOCHS = 50
# generate seeds for executions
SEEDS = [randint(1, 100) for _ in range(EXECUTIONS)]
MODEL_SEED = randint(1, 100)

EPSILON = 0.00000011920928955078125
torch.set_default_dtype(torch.float32)

# Get cpu, gpu or mps device for training.
device = (
    "cuda:1"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def test_activations(model, results_path):
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9)
    print("#"*30)
    print(f"CREATING RESULTS FOR {results_path}:")
    print("#"*30)
    tester.run(dataloaders, model, optimizer, loss_fn, device, results_path, epochs=EPOCHS, seeds=SEEDS, relu_count=True)

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

# Define model
print(f"MODEL SEED: {MODEL_SEED}")
torch.manual_seed(MODEL_SEED)
model = model_utils.Hochuli(INPUT_SHAPE, NUM_CLASSES, nn.ReLU())

test_activations(model, "results/hochuli/original.csv")

model_utils.reset_model(model)
model_utils.replace_layers(model, [nn.ReLU], relu.ReLU6Count(EPSILON))
test_activations(model, "results/hochuli/count.csv")

model_utils.reset_model(model)
model_utils.replace_layers(model, [relu.ReLU6Count], nn.GELU())
test_activations(model, "results/hochuli/diff.csv")

print("Done!")

