import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from random import randint
import copy

import relu
import model_utils
import tester

EXECUTIONS = 5
EPOCHS = 50
# generate seeds for executions
SEEDS = [randint(1, 100) for _ in range(EXECUTIONS)]
MODEL_SEED = 42

EPSILON = 0.00000011920928955078125
torch.set_default_dtype(torch.float32)

device = input()
if not device:
    device = "cuda"
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
model = model_utils.HochuliDoubleKernels(INPUT_SHAPE, NUM_CLASSES, nn.ReLU())

model_original = copy.deepcopy(model)

model_count = copy.deepcopy(model)
model_utils.replace_layers(model_count, [nn.ReLU], relu.ReLUCount(EPSILON))

model_diff = copy.deepcopy(model)
model_utils.replace_layers(model_diff, [nn.ReLU], nn.GELU())


# warmup to reduce time discrepancy
print("WARMUP")
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9)
tester.run(dataloaders, model, optimizer, loss_fn, device, "warmup.csv", epochs=5, seeds=[0], relu_count=True)
print("END WARMUP")
print()

test_activations(model_original, "results/hochuli_double/original.csv")
test_activations(model_count, "results/hochuli_double/count.csv")
test_activations(model_diff, "results/hochuli_double/diff.csv")

print(f"MODEL SEED = {MODEL_SEED}")
print(f"SEEDS = {SEEDS}")
print("Done!")