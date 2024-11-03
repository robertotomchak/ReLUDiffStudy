import torch
import torch.nn as nn
from time import time

N = 1_000_000
EXECUTIONS = 100

def test_activation(activation, device, n=1_000_000, executions=10):
    total = 0
    values = torch.randn(size=(executions, n), device=device, requires_grad=True)
    # warmup
    activation(values)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time()
    for i in range(executions):
        ans = activation(values[i])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time() - t
    return t / executions

def test_activation_diff(activation, device, n=1_000_000, executions=10):
    total = 0
    values = torch.randn(size=(executions, n), device=device, requires_grad=True)
    # warmup
    ans = activation(values)
    ans.backward(torch.ones_like(values))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time()
    for i in range(executions):
        values[i].grad = None
        ans = activation(values[i])
        ans.backward(torch.ones_like(values[i]))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time() - t
    return t / n

def compare_activations(activation1, activation2, name1, name2, device):
    activation1.to(device)
    activation2.to(device)
    time_activation1 = test_activation(activation1, device, n=N, executions=EXECUTIONS)
    time_activation2 = test_activation(activation2, device, n=N, executions=EXECUTIONS)
    time_activation1_b = test_activation_diff(activation1, device, n=N, executions=EXECUTIONS)
    time_activation2_b = test_activation_diff(activation2, device, n=N, executions=EXECUTIONS)
    print(f"{name1} vs {name2}")
    print(f"FORWARD: {name1} is {round(time_activation1 / time_activation2, 2)}x slower than {name2}")
    print(f"BACKWARD: {name1} is {round(time_activation1_b / time_activation2_b, 2)}x slower than {name2}")
    print("-"*30)

device = (
    "cuda:1"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("TEST DESCRIPTION")
print(f"N = {N:_}")
print(f"Executions = {EXECUTIONS}")
print(f"Device = {device}")
print(f"-"*30)

compare_activations(nn.GELU(), nn.ReLU(), "GELU", "ReLU", device)
compare_activations(nn.SiLU(), nn.Hardswish(), "Swish", "Hardswish", device)
compare_activations(nn.Sigmoid(), nn.Hardsigmoid(), "Sigmoid", "HardSigmoid", device)