'''
defining some useful functions for creating, modifying and training the desired model
'''

import torch
from torch import nn
import torchvision.models as models
import torchvision

import relu

def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)
            
        if any([isinstance(module, choice) for choice in old]):
            ## simple module
            setattr(model, n, new)


def reset_model(module):
    for layer in module.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        reset_model(layer)


def change_layer(model, layer_name, new_layer):
    model._modules[layer_name] = new_layer


def create_mobile_net(activation_hardwish, activation_relu, last_output, pre_trained=False):
    if pre_trained:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    else:
        model = models.mobilenet_v3_small()
    replace_layers(model, [nn.Hardswish], activation_hardwish)
    replace_layers(model, [nn.ReLU], activation_relu)
    out_layer = nn.Sequential(
        nn.Linear(in_features=576, out_features=1024, bias=True),
        activation_relu,
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1024, out_features=last_output, bias=True)
    )
    change_layer(model, "classifier", out_layer)
    return model


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    total_loss = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct, test_loss