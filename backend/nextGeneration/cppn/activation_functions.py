"""Contains activation functions for nodes of the CPPN."""
import inspect
import math
import sys
import torch

def get_all():
    """Returns all activation functions."""
    fns = inspect.getmembers(sys.modules[__name__])
    fns = [f[1] for f in fns if len(f)>1 and f[0] != "get_all"\
        and isinstance(f[1], type(get_all))]
    return fns

def identity(x):
    """Returns the input value."""
    return x

def sigmoid(x):
    """Returns the sigmoid of the input."""
    return torch.sigmoid(x)

def sigmoid_no_grad(x):
    """Returns the sigmoid of the input."""
    # Can have NaN in gradient
    return ((1.0 / (1.0 + torch.exp(-x))) - 0.5) * 2.0

def linear(x):
    """Returns the linear of the input."""
    return torch.minimum(torch.maximum(x, torch.tensor(-3.0)), torch.tensor(3.0)) / 3.0

def clip(x):
    """Returns the linear of the input."""
    return torch.clip(x, -1.0, 1.0)

def tanh(x):
    """Returns the hyperbolic tangent of the input."""
    y = torch.tanh(2.5*x)
    return y

def relu(x):
    """Returns the rectified linear unit of the input."""
    return (x * (x > 0)).to(dtype=torch.float32, device=x.device)

def tanh_sig(x):
    """Returns the sigmoid of the hyperbolic tangent of the input."""
    return sigmoid(tanh(x))

def pulse(x):
    """Return the pulse fn of the input."""
    return 2.0*(x % 1 < .5) -1.0


def hat(x):
    """Returns the hat function of the input."""
    x = 1.0 - torch.abs(x)
    x = torch.clip(x, 0.0, 1.0)
    return x

def round_activation(x):
    """return round(x)"""
    return torch.round(x)

def abs_activation(x):
    """Returns the absolute value of the input."""
    return torch.abs(x)

def sqr(x):
    """Return the square of the input."""
    return torch.square(x)

def elu(x):
    """Returns the exponential linear unit of the input."""
    return torch.where(x > 0, x, torch.exp(x) - 1,).to(torch.float32)

def sin(x):
    """Returns the sine of the input."""
    # y =  torch.sin(x*math.pi)
    y =  torch.sin(x)
    return y

def cos(x):
    """Returns the cosine of the input."""
    y =  torch.cos(x*math.pi)
    return y


# def gauss(x, mean=0.0, std=1.0):
#     """Returns the gaussian of the input."""
#     y = 2*torch.exp(-6.0 * (x-mean) ** 2/std**2)-1.0 
    # return y
def gauss(x):
    return torch.exp(-torch.pow(x, 2))

# def gauss(x, mean=0.0, std=1.0):
#     """Returns the gaussian of the input."""
#     y = 2*torch.exp(-20.0 * (x-mean) ** 2/std**2)-1.0
#     return y

# def gauss(x , mean=0.0 , sd=1.0):
#     prob_density = (torch.pi*sd) * torch.exp(-0.5*((x-mean)/sd)**2)
#     return prob_density


def triangle(X):
    return 1 - 2 *torch.arccos((1 - .0001) * torch.sin(2 * torch.pi * X))/torch.pi
def square(X):
    return 2* torch.arctan(torch.sin(2 *torch.pi* X)/.0001)/torch.pi
def sawtooth(X):
    return (1 + triangle((2*X - 1)/4.0) * square(X/2.0)) / 2.0


def softsign(x):
    return x / (1 + torch.abs(x))

def softplus(x):
    return torch.log(1 + torch.exp(x))

def tanh_softsign(x):
    return softsign(tanh(x))

def tanh_softsign_norm(x):
    return 0.5+tanh_softsign(x)



class SinActivation(torch.nn.Module):
    def __init__(self):
        super(SinActivation, self).__init__()
        return
    def forward(self, x):
        return torch.sin(x)


class CosActivation(torch.nn.Module):
    def __init__(self):
        super(CosActivation, self).__init__()
        return
    def forward(self, x):
        return torch.cos(x)


class RoundActivation(torch.nn.Module):
    def __init__(self):
        super(RoundActivation, self).__init__()
        return
    def forward(self, x):
        return torch.round(x)
    

class GaussActivation(torch.nn.Module):
    def __init__(self):
        super(GaussActivation, self).__init__()
        return
    def forward(self, x):
        return gauss(x)
    
    
class SigmoidActivation(torch.nn.Module):
    def __init__(self):
        super(SigmoidActivation, self).__init__()
        return
    def forward(self, x):
        return torch.sigmoid(x)
    
    
class TanhActivation(torch.nn.Module):
    def __init__(self):
        super(TanhActivation, self).__init__()
        return
    def forward(self, x):
        return torch.tanh(x)
    

class IdentityActivation(torch.nn.Module):
    def __init__(self):
        super(IdentityActivation, self).__init__()
        return
    def forward(self, x):
        return x

class LinearActivation(torch.nn.Module):
    # todo: add bias?
    def __init__(self):
        super(LinearActivation, self).__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        return
    def forward(self, x):
        return linear(x) + self.bias


class SoftPlusActivation(torch.nn.Module):
    def __init__(self):
        super(SoftPlusActivation, self).__init__()
        return
    def forward(self, x):
        return torch.log(1 + torch.exp(x))



class BaseConvActivation(torch.nn.Module):
    def __init__(self):
        super(BaseConvActivation, self).__init__()
        
    def forward(self, x):
        x = x.unsqueeze(0)
        return self.activation(self.conv(x)).squeeze(0)


class Conv3x3Activation(torch.nn.Module):
    def __init__(self):
        super(Conv3x3Activation, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.activation = torch.nn.Identity()
    def forward(self, x):
        x = x.unsqueeze(0)
        return self.activation(self.conv(x)).squeeze(0)


class Conv5x5Activation(BaseConvActivation):
    def __init__(self):
        super(Conv5x5Activation, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, 5, padding=2, bias=False)
        self.activation = torch.nn.Identity()
    def forward(self, x):
        x = x.unsqueeze(0)
        return self.activation(self.conv(x)).squeeze(0)
   

class Conv7x7Activation(BaseConvActivation):
    def __init__(self):
        super(Conv7x7Activation, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, 7, padding=3, bias=False)
        self.activation = torch.nn.Identity()
    def forward(self, x):
        x = x.unsqueeze(0)
        return self.activation(self.conv(x)).squeeze(0)
   
    
class Conv9x9Activation(BaseConvActivation):
    def __init__(self):
        super(Conv9x9Activation, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, 9, padding=4, bias=False)
        self.activation = torch.nn.Identity()
    def forward(self, x):
        x = x.unsqueeze(0)
        return self.activation(self.conv(x)).squeeze(0)


class KernelSharpenActivation(BaseConvActivation):
    def __init__(self):
        super(KernelSharpenActivation, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.conv.weight.data = torch.tensor([[[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]]], dtype=torch.float32)
        self.conv.requires_grad_(False)
        self.activation = torch.nn.Identity()
    def forward(self, x):
        x = x.unsqueeze(0)
        return self.activation(self.conv(x)).squeeze(0)
    
    
class KernelBlurActivation(BaseConvActivation):
    def __init__(self):
        super(KernelBlurActivation, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.conv.weight.data = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]], dtype=torch.float32) / 16.0
        self.conv.requires_grad_(False)
        self.activation = torch.nn.Identity()
    def forward(self, x):
        x = x.unsqueeze(0)
        return self.activation(self.conv(x)).squeeze(0)


class KernelEdgeActivation(BaseConvActivation):
    def __init__(self):
        super(KernelEdgeActivation, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.conv.weight.data = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=torch.float32)
        self.conv.requires_grad_(False)
        self.activation = torch.nn.Identity()
    def forward(self, x):
        x = x.unsqueeze(0)
        return self.activation(self.conv(x)).squeeze(0)
    
    
class KernelEmbossActivation(BaseConvActivation):
    def __init__(self):
        super(KernelEmbossActivation, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.conv.weight.data = torch.tensor([[[[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]]], dtype=torch.float32)
        self.conv.requires_grad_(False)
        self.activation = torch.nn.Identity()
    def forward(self, x):
        x = x.unsqueeze(0)
        return self.activation(self.conv(x)).squeeze(0)


class NormalizeActivation(torch.nn.Module):
    def __init__(self):
        super(NormalizeActivation, self).__init__()
        return
    def forward(self, x):
        range = torch.max(x) - torch.min(x)
        if range == 0:
            return x
        return (x - torch.min(x)) / range
    
class NormalizeNeg1To1Activation(torch.nn.Module):
    def __init__(self):
        super(NormalizeNeg1To1Activation, self).__init__()
        return
    def forward(self, x):
        range = torch.max(x) - torch.min(x)
        if range == 0:
            return x
        x = (x - torch.min(x)) / range
        return x * 2.0 - 1.0
    
class StandardizeActivation(torch.nn.Module):
    def __init__(self):
        super(StandardizeActivation, self).__init__()
        return
    def forward(self, x):
        mean = torch.mean(x)
        std = torch.std(x)
        if std == 0:
            return x
        return (x - mean) / std


TORCH_ACTIVATION_FUNCTIONS={
    "Tanh": torch.nn.Tanh,
    "ReLU": torch.nn.ReLU,
    "Sigmoid": torch.nn.Sigmoid,
    "Identity": torch.nn.Identity,
    "SoftPlus": torch.nn.Softplus,
    "SoftSign": torch.nn.Softsign,
    "ELU": torch.nn.ELU,
    "SELU": torch.nn.SELU,
    "GLU": torch.nn.GLU,
    "LeakyReLU": torch.nn.LeakyReLU,
    "LogSigmoid": torch.nn.LogSigmoid,
    "Hardshrink": torch.nn.Hardshrink,
    "Hardtanh": torch.nn.Hardtanh,
    "Hardswish": torch.nn.Hardswish,
    "LogSoftmax": torch.nn.LogSoftmax,
    "PReLU": torch.nn.PReLU,
    "RReLU": torch.nn.RReLU,
    "CELU": torch.nn.CELU,
    "GELU": torch.nn.GELU,
}
    
ACTIVATION_FUNCTIONS={
    k:v for k,v in locals().items() if isinstance(v, type) and (issubclass(v, torch.nn.Module))
}
ACTIVATION_FUNCTIONS.update(TORCH_ACTIVATION_FUNCTIONS)

def register_activation_function(name, fn):
    ACTIVATION_FUNCTIONS[name] = fn