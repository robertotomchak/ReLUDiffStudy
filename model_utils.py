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


def create_mobile_net(activation_hardwish, activation_relu, activation_hardsigmoid, last_output, pre_trained=False, freeze=False):
    if pre_trained:
        model = models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)
    else:
        model = models.mobilenet_v3_small()
    if activation_hardwish:
        replace_layers(model, [nn.Hardswish], activation_hardwish)
    if activation_relu:
        replace_layers(model, [nn.ReLU], activation_relu)
    if activation_hardsigmoid:
        replace_layers(model, [nn.Hardsigmoid], activation_hardsigmoid)
        
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, last_output)
    
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier.requires_grad_ = True
    else:
        for param in model.parameters():
            param.requires_grad = True
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
