import torch
import torch.nn as nn
import time

N = 1_000
EXECUTIONS = 30
DEVICE = "cuda:1"

def print_results(name1, name2, times):
    print(f"{name1} vs {name2}")
    print(f"{name1}: {times[name1]*1000} miliseconds")
    print(f"{name2}: {times[name2]*1000} miliseconds")
    print(f"{name1} is {round(times[name1] / times[name2], 2)}x slower than {name2}")
    print("-"*30)

def test_activation_speed(input_size=(N, N), executions=EXECUTIONS, device=DEVICE):
    # Generate a random input tensor
    input_tensor = torch.randn(input_size)
    
    # List of activation functions to test
    activations = {
        "Tanh": nn.Tanh(),  # just for warmup
        "ReLU": nn.ReLU(),
        "GELU": nn.GELU(),
        "Hardswish": nn.Hardswish(),
        "Swish": nn.SiLU(),
        "Hardsigmoid": nn.Hardsigmoid(),
        "Sigmoid": nn.Sigmoid()
    }
    for name, activation in activations.items():
        activation.to(device)
    
    # Dictionary to store execution times
    execution_times = {}
    
    # Perform the benchmarking for each activation function
    for name, activation in activations.items():
        t = time.time()
        for i in range(executions):
            ans = activation(input_tensor)
        torch.cuda.current_stream().synchronize()
        t = time.time() - t
        
        avg_time = (t) / executions
        execution_times[name] = avg_time

    # Print the execution times
    print_results("GELU", "ReLU", execution_times)
    print_results("Swish", "Hardswish", execution_times)
    print_results("Sigmoid", "Hardsigmoid", execution_times)


print("TEST DESCRIPTION:")
print(f"TENSOR SIZE = ({N}, {N})")
print(f"EXECUTIONS = {EXECUTIONS}")
print(f"DEVICE = {DEVICE}")
print("-"*30)

test_activation_speed()
