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

EPSILON = 0.00000011920928955078125
torch.set_default_dtype(torch.float32)

device = input()
if not device:
    device = "cuda"

# creates the mobilenet with given activations and makes the test
def test_activations(new_relu, new_hs, new_sigmoid, results_path):
    model = model_utils.create_mobile_net(new_relu, new_hs, new_sigmoid, 
                                          10, pre_trained=True, freeze=False)
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


train_data, val_data = random_split(training_data, [0.7, 0.3])


# Create data loaders.
dataloaders = {}
dataloaders["train"] = DataLoader(train_data, batch_size=32, shuffle=True)
dataloaders["val"] = DataLoader(val_data, batch_size=1024, shuffle=False)
dataloaders["test"] = DataLoader(test_data, batch_size=1024, shuffle=False)

# test for each activation functions group
test_activations(nn.ReLU(), nn.Hardswish(), nn.Hardsigmoid(), "results/mobilenet/original.csv")
test_activations(relu.ReLUCount(EPSILON), relu.HardswishCount(EPSILON), relu.HardSigmoidCount(EPSILON), "results/mobilenet/count.csv")
test_activations(nn.GELU(), nn.SiLU(), nn.Sigmoid(), "results/mobilenet/diff.csv")
print(f"SEEDS = {SEEDS}")
print("Done!")
