import torch
import torch.nn as nn

TOTAL_RELU_CALL = 0
ZERO_RELU_CALL = 0

class ReLUCountFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, tol):
        ctx.save_for_backward(input)
        ctx.tol = tol
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        global TOTAL_RELU_CALL
        global ZERO_RELU_CALL
        TOTAL_RELU_CALL += torch.numel(grad_input)
        ZERO_RELU_CALL += torch.numel(grad_input[input.abs() <= ctx.tol])

        grad_input[input < 0] = 0
        grad_input[input == 0] = 0
        return grad_input, None
    
class ReLUCount(nn.Module):
    def __init__(self, tol):
        super(ReLUCount, self).__init__()
        self.tol = tol

    def forward(self, input):
        return ReLUCountFunction.apply(input, self.tol)
    

class ReLU6CountFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, tol):
        ctx.save_for_backward(input)
        ctx.tol = tol
        return input.clamp(min=0, max=6)
    
    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        global TOTAL_RELU_CALL
        global ZERO_RELU_CALL
        TOTAL_RELU_CALL += torch.numel(grad_input)
        ZERO_RELU_CALL += torch.numel(grad_input[input.abs() <= ctx.tol])
        ZERO_RELU_CALL += torch.numel(grad_input[(input - 6).abs() <= ctx.tol])

        grad_input[input < 0] = 0
        grad_input[input == 0] = 0
        grad_input[input > 6] = 0
        grad_input[input == 6] = 0
        return grad_input, None
    
class ReLU6Count(nn.Module):
    def __init__(self, tol):
        super(ReLU6Count, self).__init__()
        self.tol = tol

    def forward(self, input):
        return ReLU6CountFunction.apply(input, self.tol)
    

class HardSigmoidCountFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, tol):
        ctx.save_for_backward(input)
        ctx.tol = tol
        output = input.clone()
        output[input <= -3] = 0
        output[input >=  3] = 1
        output[input.abs() < 3] = input[input.abs() < 3] / 6 + 0.5
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        global TOTAL_RELU_CALL
        global ZERO_RELU_CALL
        TOTAL_RELU_CALL += torch.numel(grad_input)
        ZERO_RELU_CALL += torch.numel(grad_input[((input).abs() - 3).abs() <= ctx.tol])

        grad_input[input < -3] = 0
        grad_input[input == -3] = 0
        grad_input[input > 3] = 0
        grad_input[input == 3] = 0
        grad_input[input.abs() < 3] = 1 / 6
        return grad_input, None

class HardSigmoidCount(nn.Module):
    def __init__(self, tol):
        super(HardSigmoidCount, self).__init__()
        self.tol = tol

    def forward(self, input):
        return HardSigmoidCountFunction.apply(input, self.tol)


class HardswishCount(nn.Module):
    def __init__(self, tol):
        super(HardswishCount, self).__init__()
        self.tol = tol

    def forward(self, input):
        return input * ReLU6CountFunction.apply(input+3, self.tol) / 6
    

def restart_count():
    global TOTAL_RELU_CALL
    global ZERO_RELU_CALL
    TOTAL_RELU_CALL = 0
    ZERO_RELU_CALL = 0


def get_relu_proportion():
    global TOTAL_RELU_CALL
    global ZERO_RELU_CALL
    return ZERO_RELU_CALL / TOTAL_RELU_CALL
