'''
    Makes the test for any given model
'''

import torch
import torchvision
import torch.nn as nn
import time
import numpy as np

import model_utils
import relu

def run(dataloader, model, old_activations, new_activation, optimizer, loss, device, output_layer=None, executions=5, epochs=10):
    data_train = dataloader["train"]
    data_val = dataloader["val"]
    data_test = dataloader["test"]

    model_utils.replace_layers(model, old_activations, new_activation)
    if output_layer:
            model.fc = output_layer

    for i in range(executions):
        model_utils.reset_model(model)
        print(f"EXECUTION {i+1}")
        train_loss = []
        val_loss = []
        val_acc = []
        start = time.time()
        for j in range(epochs):
            print("-"*30)
            print(f"epoch {j+1}")
            train_loss.append(model_utils.train(data_train, model, loss, optimizer, device))
            acc, loss_score = model_utils.test(data_val, model, loss, device)
            val_loss.append(loss_score)
            val_acc.append(acc)
        end = time.time()
        print(train_loss[-1])
        print(val_loss[-1])
        print(val_acc[-1])
        if isinstance(new_activation, relu.ReLUCount):
            print(f"Relu proportion: {relu.get_relu_proportion()} [{relu.ZERO_RELU_CALL} / {relu.TOTAL_RELU_CALL}]")
            relu.restart_count()
        print(f"Time: {end - start} seconds")


