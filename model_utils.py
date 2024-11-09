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
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True
    return model


# calculates size of output of convolutions of the Hachuli layer
def size_conv_output(input_shape):
    w, h = input_shape[1], input_shape[2]
    # first pool layer
    w, h = w // 2, h // 2
    # second conv
    w, h = w - 2, h - 2
    # second pool
    w, h = w // 2, h // 2
    # third conv
    w, h = w - 2, h - 2
    # third pool
    w, h = w // 2, h // 2
    return w * h * 64


# defines the hochuli network
class Hochuli(nn.Module):
    def __init__(self, input_shape, last_output, activation_relu):
        super(Hochuli, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1),
            activation_relu,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            activation_relu,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            activation_relu,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.flatten = nn.Flatten()
        self.fully_connected = nn.Sequential(
            nn.Linear(size_conv_output(input_shape), 64), 
            activation_relu,
            nn.Linear(64, last_output)
        )

    def forward(self, x):
        x = self.convolution(x)
        x = self.flatten(x)
        x = self.fully_connected(x)
        return x


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
