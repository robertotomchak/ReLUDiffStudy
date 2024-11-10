'''
    Makes the test for any given model
'''

import torch
import torchvision
import torch.nn as nn
import time
import numpy as np
import pandas as pd
import random

import model_utils
import relu


def run(dataloader, model, optimizer, loss, device, csv_file, seeds, epochs=10, relu_count=False):
    TEST_INDEX = -1
    executions = len(seeds)
    columns = ["execution", "epoch", "train_loss", "val_loss", "val_acc", "time"]
    if relu_count:
        columns.append("zero_relu_call")
        columns.append("total_relu_call")
        relu.restart_count()
    final_data = {column: [] for column in columns}

    data_train = dataloader["train"]
    data_val = dataloader["val"]
    data_test = dataloader["test"]

    for i in range(executions):
        torch.manual_seed(seeds[i])
        random.seed(seeds[i])
        model_utils.reset_model(model)
        print(f"EXECUTION {i+1}, SEED {seeds[i]}")
        # train
        for j in range(epochs):
            print("-"*30)
            print(f"EPOCH {j+1}")
            start = time.time()
            train_loss = (model_utils.train(data_train, model, loss, optimizer, device))
            val_acc, val_loss = model_utils.test(data_val, model, loss, device)
            end = time.time()

            final_data["execution"].append(i+1)
            final_data["epoch"].append(j+1)
            final_data["train_loss"].append(train_loss)
            final_data["val_loss"].append(val_loss)
            final_data["val_acc"].append(val_acc)
            final_data["time"].append(end - start)
            if relu_count:
                final_data["zero_relu_call"].append(relu.ZERO_RELU_CALL)
                final_data["total_relu_call"].append(relu.TOTAL_RELU_CALL)
                relu.restart_count()
        # test
        print("-"*30)
        print("TEST")
        print("-"*30)
        start = time.time()
        val_acc, val_loss = model_utils.test(data_test, model, loss, device)
        torch.cuda.current_stream().synchronize()  # Waits for everything to finish running
        end = time.time()
        final_data["execution"].append(i+1)
        final_data["epoch"].append(TEST_INDEX)
        final_data["train_loss"].append(np.nan)
        final_data["val_loss"].append(val_loss)
        final_data["val_acc"].append(val_acc)
        final_data["time"].append(end - start)
        if relu_count:
            final_data["zero_relu_call"].append(relu.ZERO_RELU_CALL)
            final_data["total_relu_call"].append(relu.TOTAL_RELU_CALL)
            relu.restart_count()

    # save to csv
    df = pd.DataFrame(final_data)
    df.to_csv(csv_file, index=False)

