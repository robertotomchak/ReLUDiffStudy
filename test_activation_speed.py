import torch
import torch.nn as nn
from time import time

N = 10_000_000
EXECUTIONS = 100

def test_activation(activation1, activation2, device, N=1_000_000, executions=10):
    total1, total2 = 0, 0
    for _ in range(executions):
        values = torch.rand(N, device=device)
        time1 = time()
        ans1 = activation1(values)
        time1 = time() - time1

        time2 = time()
        ans2 = activation2(values)
        time2 = time() - time2

        total1 += time1
        total2 += time2
    return total1 / executions, total2 / executions

device = (
    "cuda"
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


relu = nn.ReLU()
gelu = nn.GELU()
relu.to(device)
gelu.to(device)

time_relu, time_gelu = test_activation(relu, gelu, device, N=N, executions=EXECUTIONS)
print("ReLU vs GELU")
print(f"ReLU: {time_relu} seconds")
print(f"GELU: {time_gelu} seconds")
diff = (time_gelu - time_relu) / time_relu
print(f"GELU is {round(100*diff, 2)}% slower than ReLU")
print("-"*30)


hardswish = nn.Hardswish()
swish = nn.SiLU()
hardswish.to(device)
swish.to(device)

time_hs, time_s = test_activation(relu, gelu, device, N=N, executions=EXECUTIONS)
print("Hardswish vs Swish")
print(f"Hardswish: {time_hs} seconds")
print(f"Swish: {time_s} seconds")
diff = (time_s - time_hs) / time_hs
print(f"Swish is {round(100*diff, 2)}% slower than Harswish")
print("-"*30)