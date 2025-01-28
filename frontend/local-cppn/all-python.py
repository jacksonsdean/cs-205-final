"""Contains activation functions for nodes of the CPPN."""
import inspect
import math
import sys
import torch
import numpy as np

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
    
# name to function.py

from typing import Callable
name_to_fn = ACTIVATION_FUNCTIONS

def register_activation_function(name, fn):
    """
    Registers a function as an activation function.
    params:
        name: The name of the function.
        fn: The function.
    """
    ACTIVATION_FUNCTIONS[name] = fn


# config.py
 
    
"""Stores configuration parameters for the CPPN."""
import inspect
import json
from typing import Callable
import imageio.v2 as iio
import torch
import logging


class CPPNConfig:
    """Stores configuration parameters for the CPPN."""
    version = [2, 0, 0]
    
    # pylint: disable=too-many-instance-attributes
    def __init__(self, file=None) -> None:
        # Initialize to default values
        # These are only used if a sub-class does not override them
        self.seed = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32
        
        self.activation_mode = "node" # "node", "layer", or "population" 
        
        self.num_upsamples = 0
        self.num_conv=0
        self.num_post_conv=0
        self.dry_run = False
        self.res_w = 128
        self.res_h = 128
        self.color_mode = "RGB" # HSL, RGB, or L
        
        self.save_h=1024
        self.save_w=1024
        
        self.init_connection_probability = 0.5
        self.init_connection_probability_fourier = 0.1
        self.fourier_seed = 'random'
        
        # self.force_init_path_inputs_outputs = True
        # self.dense_init_connections = False
        # self.activations = [sin, cos, gauss, linear, tanh]
        self.activations = [SinActivation, CosActivation, GaussActivation, IdentityActivation, TanhActivation]
        self.normalize_outputs = False # None, "picbreeder", "sigmoid", 'min_max', 'abs_tanh'
        self.node_agg = 'sum'
        
        # self.output_blur = 0.0 # don't blur
        
        self.genome_type = None # algorithm default
        
        self.single_structural_mutation = False
        
        """DGNA: the probability of adding a node is 0.5 and the
        probability of adding a connection is 0.4.
        SGNA: probability of adding a node is 0.05 and the
         probability of adding a connection is 0.04.
        NEAT: probability of adding a node is 0.03 and the
          probability of adding a connection is 0.05."""
        self.prob_mutate_activation = .15
        self.prob_add_connection = .15 # 0.05 in the original NEAT
        self.prob_add_node = .15 # 0.03 in original NEAT
        self.prob_remove_node = 0.05
        self.prob_disable_connection = .05
        
        self.topology_mutation_iters=1
        self.connection_bloat=0
        self.allow_recurrent=False # Not supported
        
        # use 0 with SGD:
        self.prob_mutate_weight = 0.0 # .80 in the original NEAT
        self.prob_weight_reinit = 0.0 # .1 in the original NEAT (.1 of .8)
        self.prob_mutate_bias = 0.0
        self.prob_mutate_response = 0.0
        
        self.bias_mutation_std = 0.0
        self.weight_mutation_std = 1.0
        
        self.connection_prune_threshold = 0.0
        
        self.bloat_prune_rate = 0.0
        self.extra_bloat = 0
        
        self.do_crossover = True
        
        self.weight_init_std = 3.0
        self.weight_threshold = 0
        self.prob_random_restart =.001
        self.prob_reenable_connection = 0.95
        self.coord_range = (-0.5, 0.5)
        self.output_activation = IdentityActivation
        self.target_resize = None # use original size
        
        self.output_dir = None
        self.experiment_condition = "_default"

        # DGNA/SGMA uses 1 or 2 so that patterns in the initial
        # generation would be nontrivial (Stanley, 2007).
        # Original NEAT paper uses 0
        self.hidden_nodes_at_start = 0

        self.allow_input_activation_mutation = True

        self.animate = False
        
        self.with_grad = True
        self.sgd_learning_rate = 2.0
        self.batch_lr_mod = False # don't change sgd lr based on batch size
        self.prob_sgd_weight = 1.0 # update all weights
        self.sgd_early_stop_delta = -0.0005
        self.sgd_l2_reg = 0.0 # don't use L2 regularization
        self.sgd_steps = 10
        self.no_param_mutations = False
        self.sgd_clamp_grad = False
        self.sgd_every = 1
        self.sgd_early_stop = 5
        self.mutate_sgd_lr_sigma = self.sgd_learning_rate * 0.01
        self.max_weight = 10.0
        
        self.resume=None # run to resume from (path to checkpoints folder)
        
        # Fourier features:
        self.use_fourier_features = True
        self.n_fourier_features = 0 # TODO add to settings, but TODO is bugged (seed changes every gen)
        self.fourier_feature_scale = 1.0
        self.fourier_mult_percent = 0.05
        self.fourier_sin_and_cos = True

        # https://link.springer.com/content/pdf/10.1007/s10710-007-9028-8.pdf page 148
        self.use_input_bias = False # SNGA,
        self.use_radial_distance = True # bias towards radial symmetry
        
        self.num_inputs = 2 + self.n_fourier_features
        # self.num_extra_inputs = 0 # e.g. for latent vector
        self.num_outputs = len(self.color_mode)
        if self.use_input_bias:
            self.num_inputs += 1
        if self.use_radial_distance:
            self.num_inputs += 1
        
        # guarantee that the inputs connect to the outputs
        self.force_init_path_inputs_outputs = True
        
        self._make_dirty()
        
        self.file = file
        
        if self.file is not None:
            import json
            print(f"Loading config from {self.file}")
            with open(self.file, "r") as f:
                loaded = json.load(f)
                use = loaded
                if "controls" in loaded:
                    use = loaded["controls"]
                self.from_json(use, print_out=True)
                f.close()
        
    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key != 'dirty':
            self._make_dirty()
            
            
    def _make_dirty(self):
        self.dirty = True

    def _not_dirty(self):
        self.dirty = False

        
    def apply(self, parameter, value):
        """Applies a parameter value to the configuration."""
        setattr(self, parameter, value)
        
    def set_res(self, *res):
        """Sets the resolution of the output image."""
        if len(res) == 1:
            self.res_w = res[0]
            self.res_h = res[0]
        else:
            self.res_w = res[0]
            self.res_h = res[1]
    
    def clone(self):
        return self.__class__.create_from_json(self.to_json(), self.__class__)
    
    #################    
    # Serialization #
    #################
    
    def serialize(self):
        self.fns_to_strings()
        
    def deserialize(self):
        self.strings_to_fns()

    def fns_to_strings(self):
        """Converts the activation functions to strings."""
        if self.genome_type and not isinstance(self.genome_type, str):
            self.genome_type = self.genome_type.__name__
        self.device = str(self.device)
        self.activations= [fn.__name__ if (not isinstance(fn, str) and not fn is None) else fn for fn in self.activations]
        if hasattr(self, "fitness_function") and isinstance(self.fitness_function, Callable):
            self.fitness_function = self.fitness_function.__name__
        
        if self.output_activation is None:
            self.output_activation = ""
        else:
            self.output_activation = self.output_activation.__name__ if\
                not isinstance(self.output_activation, str) else self.output_activation

        if hasattr(self, "target_name") and self.target_name is not None:
            self.target = self.target_name
        
        self.dtype = str(self.dtype) # TODO deserialize 

    def strings_to_fns(self):
        """Converts the activation functions to functions."""
        if self.genome_type:
            found = False
            modules = sys.modules
            for m in modules:
                try:
                    for c in inspect.getmembers(m, inspect.isclass):
                        if c[0] == self.genome_type:
                            self.genome_type = c[1]
                            found = True
                            break
                    if found:
                        break
                except:
                    continue
                
        self.device = torch.device(self.device)
        self.activations = [name_to_fn[name] if isinstance(name, str) else name for name in self.activations ]
        if hasattr(self, "target") and isinstance(self.target, str):
            try:
                self.target = torch.tensor(iio.imread(self.target), dtype=torch.float32, device=self.device)
            except FileNotFoundError:
                self.target = None
        # try:
        #     if hasattr(self, "fitness_function") and isinstance(self.fitness_function, str):
        #         self.fitness_function = name_to_fn[self.fitness_function]
        # except ValueError:
        #     self.fitness_function = None
        self.output_activation = name_to_fn[self.output_activation] if (isinstance(self.output_activation, str) and len(self.output_activation)>0) else self.output_activation
        if self.output_activation == "":
            self.output_activation = None
       
        # if isinstance(self.dtype, str):
            # self.dtype = getattr(torch, self.dtype.removeprefix("torch."))
        
        if hasattr(self, "fitness_schedule") and self.fitness_schedule is not None:
            for i, fn in enumerate(self.fitness_schedule):
                if isinstance(fn, str):
                    self.fitness_schedule[i] = name_to_fn(fn)

    def to_json(self):
        """Converts the configuration to a json string."""
        self.fns_to_strings()
        data = self.__dict__.copy()
        data['version'] = self.version
        json_string = json.dumps(data, sort_keys=True, indent=4)
        self.strings_to_fns()
        return json_string


    def from_json(self, json_dict, print_out=False, warn=True):
        """Converts the configuration from a json string."""
        if isinstance(json_dict, dict):
            json_dict = json.dumps(json_dict)
            json_dict = json.loads(json_dict)
        elif isinstance(json_dict, str):
            json_dict = json.loads(json_dict)
        self.fns_to_strings()
        self.version = json_dict['version']
        for key, value in json_dict.items():
            if print_out:
                print(f"Setting {key} to {value}")
            if not key in self.__dict__ and warn:
                logging.warning(f"Unexpected key {key} in config {self.__class__.__name__}")
            setattr(self, key, value)
        self.strings_to_fns()
        
    def save(self, filename):
        """Saves the configuration to a file."""
        with open(filename, "w") as f:
            f.write(self.to_json())
            f.close()
            
    def load_saved(self, filename,print_out=False):
        with open(filename, 'r') as infile:
            self.from_json(infile.read(), print_out)
            infile.close()
            
    @staticmethod
    def create_from_json(json_str, config_type=None, warn=True):
        """Creates a configuration from a json string."""
        if config_type is None:
            config_type = CPPNConfig
        config = config_type()
        if isinstance(json_str, str):
            json_str = json.loads(json_str)
        config.version = json_str['version']
        for key, value in json_str.items():
            if not key in config.__dict__ and warn:
                logging.warning(f"Unexpected key {key} in config {config_type}")
            setattr(config, key, value)
        config.strings_to_fns()
        
        config.num_inputs = 2 + config.n_fourier_features
        # config.num_extra_inputs = 0 # e.g. for latent vector
        config.num_outputs = len(config.color_mode)
        if config.use_input_bias:
            config.num_inputs += 1
        if config.use_radial_distance:
            config.num_inputs += 1
        return config


    def get(self, name, default=None):
        return self.__dict__.get(name, default)
    
    
    
    
    
# graph_util.py
    
    
    
    
    
"""Contains utility functions"""
import inspect
import random
import sys



def is_valid_connection(from_node, to_node, config):
    """
    Checks if a connection is valid.
    params:
        from_node: The node from which the connection originates.
        to_node: The node to which the connection connects.
        config: The settings to check against
    returns:
        True if the connection is valid, False otherwise.
    """
    if from_node.layer == to_node.layer:
        return False  # don't allow two nodes on the same layer to connect

    if not config.allow_recurrent and from_node.layer > to_node.layer:
        return False  # invalid

    return True


def name_to_fn(name) -> callable:
    """
    Converts a string to a function.
    params:
        name: The name of the function.
    returns:
        The function.
    """
    assert isinstance(name, str), f"name must be a string but is {type(name)}"
    if name == "":
        return None
    fns = inspect.getmembers(sys.modules[__name__])
    return fns[[f[0] for f in fns].index(name)][1]


def choose_random_function(config) -> callable:
    """Chooses a random activation function from the activation function module."""
    random_fn = random.choice(config.activations)
    return random_fn


def get_disjoint_connections(this_cxs, other_innovation):
    """returns connections in this_cxs that do not share an innovation number with a
        connection in other_innovation"""
    if(len(this_cxs) == 0 or len(other_innovation) == 0):
        return []
    return [t_cx for t_cx in this_cxs if\
        (t_cx.innovation not in other_innovation and t_cx.innovation < other_innovation[-1])]


def get_excess_connections(this_cxs, other_innovation):
    """returns connections in this_cxs that share an innovation number with a
        connection in other_innovation"""
    if(len(this_cxs) == 0 or len(other_innovation) == 0):
        return []
    return [t_cx for t_cx in this_cxs if\
            (t_cx.innovation not in other_innovation and t_cx.innovation > other_innovation[-1])]


def get_matching_connections(cxs_1, cxs_2):
    """returns connections in cxs_1 that share an innovation number with a connection in cxs_2
       and     connections in cxs_2 that share an innovation number with a connection in cxs_1"""
    return sorted([c1 for c1 in cxs_1 if c1.innovation in [c2.innovation for c2 in cxs_2]],
                    key=lambda x: x.innovation),\
                    sorted([c2 for c2 in cxs_2 if c2.innovation in [c1.innovation for c1 in cxs_1]],
                    key=lambda x: x.innovation)


def find_node_with_id(nodes, node_id):
    """Returns the node with the given id from the list of nodes"""
    for node in nodes:
        if node.id == node_id:
            return node
    return None


def get_ids_from_individual(individual):
    """Gets the ids from a given individual

    Args:
        individual (CPPN): The individual to get the ids from.

    Returns:
        tuple: (inputs, outputs, connections) the ids of the CPPN's nodes
    """
    inputs = [n.id for n in individual.input_nodes]
    outputs = [n.id for n in individual.output_nodes]
    connections = [(c.split(',')[0], c.split(',')[1])
                   for c in individual.enabled_connections]
    return inputs, outputs, connections


def get_candidate_nodes(s, connections):
    """Find candidate nodes c for the next layer.  These nodes should connect
    a node in s to a node not in s."""
    return set(b for (a, b) in connections if a in s and b not in s)


def get_incoming_connections(individual, node):
    """Given an individual and a node, returns the connections in individual that end at the node"""
    return list(filter(lambda x, n=node: x.to_node.id == n.id,
               individual.enabled_connections()))  # cxs that end here


# Functions below are modified from other packages
# This is necessary because AWS Lambda has strict space limits,
# and we only need a few methods, not the entire packages.

###############################################################################################
# Functions below are from the NEAT-Python package https://github.com/CodeReclaimers/neat-python/

def required_for_output(inputs, outputs, connections):
    """
    Collect the nodes whose state is required to compute the final network output(s).
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    NOTE: It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.

    Returns a set of identifiers of required nodes.
    From: https://neat-python.readthedocs.io/en/latest/_modules/graphs.html
    """

    required = set(outputs)
    s = set(outputs)
    while 1:
        # Find nodes not in S whose output is consumed by a node in s.
        t = set(a for (a, b) in connections if b in s and a not in s)

        if not t:
            break

        layer_nodes = set(x for x in t if x not in inputs)
        if not layer_nodes:
            break

        required = required.union(layer_nodes)
        s = s.union(t)

    return required


def feed_forward_layers(individual):
    """
    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    :param inputs: list of the network input nodes
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.

    Returns a list of layers, with each layer consisting of a set of node identifiers.
    Note that the returned layers do not contain nodes whose output is ultimately
    never used to compute the final network output.

    Modified from: https://neat-python.readthedocs.io/en/latest/_modules/graphs.html
    """

    inputs, outputs, connections = get_ids_from_individual(individual)

    required = required_for_output(inputs, outputs, connections)

    layers = []
    s = set(inputs)
    while 1:

        c = get_candidate_nodes(s, connections)
        # Keep only the used nodes whose entire input set is contained in s.
        t = set()
        for n in c:
            if n in required and all(a in s for (a, b) in connections if b == n):
                t.add(n)

        if not t:
            break

        layers.append(t)
        s = s.union(t)

    return layers




###############################################################################################
# Functions below are from the Scikit-image package https://scikit-image.org/docs/stable

# Copyright (C) 2019, the scikit-image team



def hsv2rgb(hsv):
    """HSV to RGB color space conversion.
    Modified  from:
    https://scikit-image.org/docs/stable/api/skimage.color.html?highlight=hsv2rgb#skimage.color.hsv2rgb

    Parameters
    ----------
    hsv : (..., 3) array_like
        The image in HSV format. Final dimension denotes channels.

    Returns
    -------
    out : (..., 3) ndarray
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `hsv` is not at least 2-D with shape (..., 3).

    Notes
    -----
    Conversion between RGB and HSV color spaces results in some loss of
    precision, due to integer arithmetic and rounding [1]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/HSL_and_HSV

    Examples
    --------
    >>> from skimage import data
    >>> img = data.astronaut()
    >>> img_hsv = rgb2hsv(img)
    >>> img_rgb = hsv2rgb(img_hsv)
    """
    arr = hsv
    hi = np.floor(arr[..., 0] * 6)
    f = arr[..., 0] * 6 - hi
    p = arr[..., 2] * (1 - arr[..., 1])
    q = arr[..., 2] * (1 - f * arr[..., 1])
    t = arr[..., 2] * (1 - (1 - f) * arr[..., 1])
    v = arr[..., 2]

    hi = np.stack([hi, hi, hi], axis=-1).astype(np.uint8) % 6
    out = np.choose(
        hi, np.stack([np.stack((v, t, p), axis=-1),
                      np.stack((q, v, p), axis=-1),
                      np.stack((p, v, t), axis=-1),
                      np.stack((p, q, v), axis=-1),
                      np.stack((t, p, v), axis=-1),
                      np.stack((v, p, q), axis=-1)]))

    return out

# END OF CODE FROM OTHER PACKAGES



# normalization.py


import torch
from torch import nn
from typing import Optional, TypeVar, Union


available_normalizations = [
            "neat",
            "inv_neat",
            "sqr_neat",
            "clamp",
            "sigmoid",
            "sigmoid_like",
            "min_max_sigmoid_like",
            "min_max",
            "inv_min_max",
            "min_max_sqr",
            "inv_min_max_sqr",
            "min_max_channel",
            "min_max_channel_sqr",
            "inv_abs_min_max_sqr",
            "inv_abs_min_max_cube",
            "inv_abs_min_max",
            "abs_min_max",
            "abs_tanh",
            "inv_abs_tanh",
            "imagenet",
            "sigmoid_imagenet",
            "imagenet_min_max",
            "min_max_imagenet",
            "inv_abs_imagenet",
            "min_max_sqr_imagenet",
            "neat_sqr_imagenet",
            "softsign",
            "tanh_softsign"
            ]


def norm_min_max(X):
    max_value = torch.max(X)
    min_value = torch.min(X)
    image_range = max_value - min_value
    X = X - min_value
    X = X/(image_range+1e-8)
    X = torch.clamp(X, 0, 1)
    return X

def norm_neat(X):
    """http://eplex.cs.ucf.edu/papers/stanley_gpem07.pdf
    "function outputs range between [−1, 1]. However, ink level is
    darker the closer the output is to zero. Therefore, an output of
    either -1 or 1 produces white." """
    X = torch.abs(X)
    X = torch.clamp(X, 0, 1)
    return X

def norm_sigmoid_like(X):
    a = 2.4020563531719796
    X = -.5*torch.erf(X/a) + .5
    return X

def norm_min_max_sigmoid_like(X):
    X = norm_min_max(X)
    X = X*2 - 1 # center around 0
    return norm_sigmoid_like(X)

def norm_tanh(X,a,b,c):
    X = torch.tanh(a*X)
    X = b+c*torch.abs(X)
    X = torch.clamp(X, 0, 1)
    return X

def norm_min_max_channel(X):
    max_value = torch.max(X, dim=1, keepdim=True)[0]
    min_value = torch.min(X, dim=1, keepdim=True)[0]
    image_range = max_value - min_value
    X = X - min_value
    X = X/(image_range+1e-8)
    X = torch.clamp(X, 0, 1)
    return X


def norm_softsign(X):
    return 0.5 + X/(1+torch.abs(X))

def norm_tanh_softsign(X):
    return  norm_softsign(torch.tanh(X))
    

class Normalization(nn.Module):
    def __init__(self, device, mean=None, std=None):
        super().__init__()
        if mean is None:
            imagenet_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
            self.mean = imagenet_normalization_mean
        else:
            self.mean = mean
        if std is None:
            imagenet_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
            self.std = imagenet_normalization_std
        else:
            self.std = std
        self.mean = torch.as_tensor(self.mean).view(-1, 1, 1)
        self.std = torch.as_tensor(self.std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

    def to(self: torch.nn.Module, device: Optional[Union[int, torch.device]] = ..., dtype: Optional[Union[torch.dtype, str]] = ...,
           non_blocking: bool = ...) -> torch.nn.Module:
        super().to(device, dtype, non_blocking)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


def handle_normalization(X, norm, imagenet_norm=None):
    assert callable(norm) or norm in available_normalizations, f"Unknown normalize_outputs value {norm}"
    if norm == "neat":
        X = norm_neat(X)
    elif norm == "inv_neat":
        X = 1.0-norm_neat(X)
    elif norm == "sqr_neat":
        X = norm_neat(X**2)
    elif norm == "clamp":
        X = torch.clamp(X, 0, 1)
    elif norm == "sigmoid":
        X = torch.sigmoid(X)
    elif norm == "sigmoid_like":
        X = norm_sigmoid_like(X)
    elif norm == "min_max_sigmoid_like":
        X = norm_min_max_sigmoid_like(X)
    elif norm == 'abs_tanh':
        # https://www.desmos.com/calculator/bluf2zbj8o
        X = norm_tanh(X, 3.0, 0, 1.0)
    elif norm == 'inv_abs_tanh':
        # https://www.desmos.com/calculator/no8l6sy1hh
        X = norm_tanh(X, 3.0, 1.15, -1.15)
    elif norm == 'min_max':
       X = norm_min_max(X)
    elif norm == 'inv_min_max':
        X = 1.0-norm_min_max(X)
    elif norm == 'min_max_sqr':
        X = norm_min_max(X) ** 2
    elif norm == 'inv_min_max_sqr':
        X = 1.0-norm_min_max(X)**2
    elif norm == 'inv_abs_min_max_cube':
        X = norm_min_max(abs(X))
        X = 1.13-X**3
    elif norm == 'inv_abs_min_max_sqr':
        X = norm_min_max(abs(X))
        X = 1.26-X**2
    elif norm == 'inv_abs_min_max':
        X = norm_min_max(abs(X))
        X = 1.5-X
    elif norm == 'abs_min_max':
        X = norm_min_max(abs(X))
    elif norm == 'min_max_sqr_imagenet':
        X = norm_min_max(X) **2
        X = imagenet_norm(X)
    elif norm == 'min_max_channel':
        X = norm_min_max_channel(X)
    elif norm == 'min_max_channel_sqr':
        X = norm_min_max_channel(X)**2
    elif norm == 'imagenet':
        X = imagenet_norm(X)
        X = torch.sigmoid(X)
    elif norm == 'sigmoid_imagenet':
        X = torch.sigmoid(X)
        X = imagenet_norm(X)
    elif norm == 'min_max_imagenet':
        X = norm_min_max(X)
        X = imagenet_norm(X)
    elif norm == 'imagenet_min_max':
        X = imagenet_norm(X)
        X = norm_min_max(X)
    elif norm == 'inv_abs_imagenet':
        X = 1.0 - torch.abs(X)
        X += 0.5
        X = imagenet_norm(X)
    elif norm == "neat_sqr_imagenet":
        X = 1.0 - torch.abs(X**2)
        X = imagenet_norm(X)
        X = torch.sigmoid(X)
    elif norm == 'softsign':
        X = norm_softsign(X)
    elif norm == 'tanh_softsign':
        X = norm_tanh_softsign(X)
    else:
        # callable
        X = norm(X)
    return X





# util.py





import torch
from torch import nn

import math
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from torch import nn
import torch
from typing import List, Union
from cv2 import resize as cv2_resize
import itertools
import random
from typing import Callable, List, Union, Tuple
   
from torchvision.transforms import GaussianBlur

def print_net(individual, show_weights=False, visualize_disabled=False):
    print(f"<CPPN {individual.id}")
    print(f"nodes:")
    for k, v in individual.node_genome.items():
        print("\t",k, "\t|\t",v.layer, "\t|\t",v.activation.__name__)
    print(f"connections:")
    for k, v in individual.connection_genome.items():
        print("\t",k, "\t|\t",v.enabled, "\t|\t",v.weight)
    print(">")
  



def get_max_number_of_hidden_nodes(population):
    max = 0
    for g in population:
        if len(list(g.hidden_nodes))> max:
            max = len(list(g.hidden_nodes))
    return max

def get_avg_number_of_hidden_nodes(population):
    count = 0
    if len(population) == 0:
        return 0
    for g in population:
        count+=len(g.node_genome) - g.n_in_nodes - g.n_outputs
    return count/len(population)

def get_max_number_of_connections(population):
    max_count = 0
    for g in population:
        count = len(list(g.enabled_connections))
        if(count > max_count):
            max_count = count
    return max_count

def get_min_number_of_connections(population):
    min_count = math.inf
    for g in population:
        count = len(list(g.enabled_connections)) 
        if(count < min_count):
            min_count = count
    return min_count

def get_avg_number_of_connections(population):
    count = 0
    if len(population) == 0:
        return 0
    for g in population:
        count+=len(list(g.enabled_connections))
    return count/len(population)



def upscale_conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True, device="cpu"):
    # return ConvTranspose2d(in_channels,out_channels,kernel_size, stride=stride, padding=padding, output_padding=1,device=device)
    layer = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, device=device, bias=bias)
   )
    return layer

def show_inputs(inputs, cols=8, cmap='viridis'):
    if not isinstance(inputs, torch.Tensor):
        # assume it's an algorithm instance
        inputs = inputs.inputs
        try:
            inputs = handle_normalization(inputs, inputs.config.normalize_outputs)
        except:
            pass # no normalization
    inputs = inputs.permute(2,0,1)
    image_grid(inputs,
               cols=cols,
               show=True,
               cmap=cmap,
               suptitle="Inputs")


def image_grid(images,
                cols=4,
                titles=None,
                show=True,
                cmap='gray',
                suptitle=None,
                title_font_size=12,
                fig_size=(10,10)):
    
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
        images = [i for i in images]
    fg = plt.figure(constrained_layout=True, figsize=fig_size)
    rows = 1 + len(images) // cols
    for i, img in enumerate(images):
        ax = fg.add_subplot(rows, cols, i + 1)
        ax.axis('off')
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        if titles is not None:
            ax.set_title(titles[i])
    if suptitle is not None:
        fg.suptitle(suptitle, fontsize=title_font_size)
    if show:
        fg.show()
    else:
        return plt.gcf()


def custom_image_grid(images:Union[torch.Tensor, np.ndarray, List[torch.Tensor]],
               cols=8, titles=None, show=True, cmap="gray"):
    assert titles is None or len(titles) == len(images)
    if isinstance(images, List):
        images = torch.stack(images).detach().cpu()
    elif isinstance(images, np.ndarray):
        ...
    elif isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    
    num = images.shape[0]
    
    rows = math.ceil(num / cols)
    fig, axs = plt.subplots(rows, cols, constrained_layout=True, figsize=(cols*2, rows*2))
    for i, ax in enumerate(axs.flatten()):
        ax.axis("off")
        if i >= num:
            ax.imshow(np.ones((num, images.shape[1], 3)), cmap="gray")
        else:    
            ax.imshow(images[:, :, i], cmap=cmap, vmin=0, vmax=1)
            if titles is not None:
                ax.set_title(f"Input {titles[i]}")
    if show:
        fig.tight_layout()
        fig.show()
    return fig
        


def gaussian_blur(img, sigma, kernel_size=(5,5)):
    return GaussianBlur(kernel_size=kernel_size, sigma=sigma)(img)
        
        
def resize(img, size):
    return cv2_resize(img, size)

def center_crop(img, r, c):
    h, w = img.shape[:2]
    r1 = int(round((h - r) / 2.))
    c1 = int(round((w - c) / 2.))
    return img[r1:r1 + r, c1:c1 + c]

def random_uniform(generator, low=0.0, high=1.0, grad=False):
    if generator:
        return ((low - high) * torch.rand(1, device=generator.device, requires_grad=grad, generator=generator) + high)[0]
    else:
        return ((low - high) * torch.rand(1, requires_grad=grad) + high)[0]
    
def random_normal (generator=None, mean=0.0, std=1.0, grad=False):
    if generator:
        return torch.randn(1, device=generator.device, requires_grad=grad, generator=generator)[0] * std + mean
    else:
        return torch.randn(1, requires_grad=grad)[0] * std + mean


def random_choice(options, count=1, replace=False)->Union[List, torch.Tensor, str]:
    """Chooses a random option from a list of options"""
    if not replace:
        indxs = torch.randperm(len(options))[:count]
        output = []
        for i in indxs:
            output.append(options[i])
        if count == 1:
            return output[0]
        return output
    else:
        return options[torch.randint(len(options), (count,))]
 

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def initialize_inputs(res_h, res_w, use_radial_dist, use_bias, n_inputs, device, coord_range=(-.5,.5), dtype=torch.float32):
        """Initializes the pixel inputs."""
        
        if not isinstance(coord_range, tuple):
            # assume it's a single range for both x and y min and max
            coord_range = (-coord_range, coord_range)
        
        if not isinstance(coord_range[0], tuple):
            # assume it's a single range for both x and y
            coord_range_x = coord_range
            coord_range_y = coord_range
        else:
            coord_range_x, coord_range_y = coord_range
            
        # Pixel coordinates are linear within coord_range
        x_vals = torch.linspace(coord_range_x[0], coord_range_x[1], res_w, device=device,dtype=dtype)
        y_vals = torch.linspace(coord_range_y[0], coord_range_y[1], res_h, device=device,dtype=dtype)

        # initialize to 0s
        inputs = torch.zeros((res_h, res_w, n_inputs), dtype=dtype, device=device, requires_grad=False)

        # assign values:
        inputs[:, :, 0] = y_vals.unsqueeze(1).repeat(1, res_w)
        inputs[:, :, 1] = x_vals.unsqueeze(0).repeat(res_h, 1)
            
        
        if use_radial_dist:
            # d = sqrt(x^2 + y^2)
            inputs[:, :, 2] = torch.sqrt(inputs[:, :, 0]**2 + inputs[:, :, 1]**2)
        if use_bias:
            inputs[:, :, -1] = torch.ones((res_h, res_w), dtype=dtype, device=device, requires_grad=False) # bias = 1.0
        
        repeat_dims = 2 # just y, x
        if use_radial_dist:
            repeat_dims += 1 # add radial dist
        n_repeats = 0   
        for i in range(n_repeats):
            inputs  = torch.cat((inputs, inputs[:, :, :repeat_dims]), dim=2)
        
        return inputs

def initialize_inputs_from_config(config):
    return initialize_inputs(config.res_h,
                            config.res_w,
                            config.use_radial_distance,
                            config.use_input_bias,
                            config.num_inputs - config.n_fourier_features,
                            config.device,
                            config.coord_range
                            )


# FROM: https://github.com/limacv/RGB_HSV_HSL
def hsl2rgb_torch(hsl: torch.Tensor) -> torch.Tensor:
    hsl = hsl.unsqueeze(0)
    # hsl = hsl.permute(2, 0, 1).unsqueeze(0)
    hsl_h, hsl_s, hsl_l = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]
    _c = (-torch.abs(hsl_l * 2. - 1.) + 1) * hsl_s
    _x = _c * (-torch.abs(hsl_h * 6. % 2. - 1) + 1.)
    _m = hsl_l - _c / 2.
    idx = (hsl_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsl)
    _o = torch.zeros_like(_c)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    rgb = rgb.squeeze(0)#.permute(1, 2, 0)
    return rgb











# fourier_features.py





# from https://github.com/tancik/fourier-feature-networks
import torch
import numpy as np
def apply_mapping(x, B:torch.tensor, sin_and_cos=False):
    if B is None:
      return x
    else:
      device = x.device
      x = x.cpu().numpy()
      B = B.cpu().numpy()
      x_proj = (2.*np.pi*x) @ B.T
      if sin_and_cos:
        result = np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)
      else:
        result = np.sin(x_proj)
      return torch.tensor(result, device=device).float()

    
    # x_proj = (2.*torch.pi*x) @ B.T
    # f0 = torch.sin(x_proj)
    # # return torch.cat([f0], dim=-1)
    # f1 = torch.cos(x_proj)
    # return torch.cat([f0,f1], dim=-1)

def input_mapping(x, b_scale:float, mapping_size:int, dims:int=2, sin_and_cos=False):
    mapping_size = mapping_size // 2 if sin_and_cos else mapping_size
    B_gauss = torch.randn((mapping_size, dims), device=x.device)
    B_gauss = B_gauss * b_scale
    return apply_mapping(x, B_gauss, sin_and_cos=sin_and_cos)
  
def add_fourier_features_from_config(x, conf):
     return add_fourier_features(
            x,
            conf.n_fourier_features,
            B_scale=conf.fourier_feature_scale,
            dims=2,
            include_original=True,
            mult_percent=conf.fourier_mult_percent,
            sin_and_cos=conf.fourier_sin_and_cos,
            seed=conf.seed
            )

def add_fourier_features(x, n_features, B_scale=10.0, dims=2, include_original=False, mult_percent=.5, sin_and_cos=False, seed=None):
    assert n_features % dims == 0, "mapping_size must be divisible by dims"
    
    original_seed = torch.initial_seed()
    if seed is not None:
      print(f"using fourier seed: {seed}")
      torch.manual_seed(seed) # force reproducibility of fourier features
      
    # get first dims features
    feats = x[:,:, :dims]
    
    if mult_percent:
      orig_n_features = n_features
      n_features = orig_n_features - int(orig_n_features * mult_percent)
      
    f_feats = input_mapping(feats, B_scale, n_features, dims=dims, sin_and_cos=sin_and_cos)
    if mult_percent:
      while f_feats.shape[-1] < orig_n_features:
        two_rand = torch.randint(0, f_feats.shape[-1], (2,))
        m = f_feats[:,:, two_rand[0]] * f_feats[:, :, two_rand[1]]
        f_feats = torch.cat([f_feats, m.unsqueeze(-1)], dim=-1)
        

    if include_original:
      X = torch.cat([x, f_feats], dim=-1)
    else:
      X = f_feats

    torch.manual_seed(original_seed) # reset seed
    
    return X



# cppn.py


"""Contains the CPPN, Node, and Connection classes."""
import json
from io import BytesIO

import base64
import cv2
import torch
from torch import nn

class Node(nn.Module):
    def __init__(self, activation, id, bias=0.0):
        super().__init__()
        # self.bias = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.tensor(float(bias)))
        self.set_activation(activation)
        self.id:str = id
        self.layer = 999
    
    def set_activation(self, activation):
        
        if isinstance(activation, type):
            self.activation = activation()
        else:
            self.activation = activation
        self.activation.to(self.bias.device)
        
    def forward(self, x):
        # return self.activation(x + self.bias)
        return self.activation(x + self.bias)
    
    def to_json(self):
        return {
            "id": self.id,
            "activation": self.activation.__class__.__name__,
            "bias": self.bias.item(),
            "layer": self.layer
        }
    
    def from_json(self, json):
        if isinstance(json["activation"], str):
            json["activation"] = ACTIVATION_FUNCTIONS[json["activation"]]
        self.id = json["id"]
        self.set_activation(__dict__[json["activation"]])
        self.bias = nn.Parameter(torch.tensor(json["bias"]))
        self.layer = json["layer"]
    
    @staticmethod
    def create_from_json(json):
        if isinstance(json["activation"], str):
            json["activation"] = ACTIVATION_FUNCTIONS[json["activation"]]
        n = Node(json["activation"], json["id"], json["bias"])
        n.layer = json["layer"]
        return n
    
    
class Connection(nn.Module):
    def __init__(self, weight, enabled=True, device='cuda'):
        super().__init__()
        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight, device=device)
        self.weight = nn.Parameter(weight)
        self.enabled:bool = enabled
        
    def forward(self, x):
        return x * self.weight

    def add_to_weight(self, delta):
        self.weight = nn.Parameter(self.weight + delta)
    
    def to_json(self):
        return {
            "weight": self.weight.item(),
            "enabled": self.enabled            
        }
        
    def from_json(self, json):
        self.weight = nn.Parameter(torch.tensor(json["weight"]))
        self.enabled = json["enabled"]

    @staticmethod
    def create_from_json(json):
        return Connection(float(json["weight"]), json["enabled"])

class CPPN(nn.Module):
    """A CPPN Object with Nodes and Connections."""
    
    current_id = 1 # 0 reserved for 'random' parent
    current_node_id = 0
    
    @staticmethod
    def get_id():
        __class__.current_id += 1
        return __class__.current_id - 1
    
    @staticmethod
    def get_new_node_id():
        """Returns a new node id."""
        __class__.current_node_id += 1
        new_id = str(__class__.current_node_id-1)
        return new_id
   
    # TODO: remove deprecated (kept for compatibility with old code)
    @property
    def n_outputs(self):
        return self.n_output
    @property
    def n_in_nodes(self):
        return len(self.input_nodes)
    @property
    def node_genome(self):
        return {node_id: self.nodes[node_id].activation for node_id in self.nodes}
    @property
    def connection_genome(self):
        return {conn_key: self.connections[conn_key].weight for conn_key in self.connections}

    ### end deprecated
    
    @property
    def input_nodes(self):
        return [n for n in self.nodes.values() if n.id in self.input_node_ids]
    
    @property
    def output_nodes(self):
        return [n for n in self.nodes.values() if n.id in self.output_node_ids]
    
    @property
    def hidden_nodes(self):
        return [n for n in self.nodes.values() if n.id not in self.input_node_ids and n.id not in self.output_node_ids]
    
    def __init__(self, config:CPPNConfig=None, do_init=True):
        if config is None:
            config = CPPNConfig()
        super().__init__() # super, innit?
        
        self.selected = False # used for selection in evolution
        self.image = None
        
        self.nodes = nn.ModuleDict()  # key: node_id (string)
        self.connections = nn.ModuleDict()  # key: (from, to) (string)
        
        self.n_input = 2 + config.n_fourier_features + int(config.use_input_bias) + int(config.use_radial_distance)   

        self.n_output = config.num_outputs
        
        self.sgd_lr = config.sgd_learning_rate
        self.parents = (-1, -1)
        self.age = 0
        self.lineage = []
        self.cell_lineage = []
        self.n_cells = 0
        self.device = config.device
        self.id = -1
        
        self.node_states = {}
        
        self.input_node_ids =   [str(i) for i in range(-1, -self.n_input - 1, -1)]
        self.output_node_ids =  [str(i) for i in range(-self.n_input - 1, -self.n_input - self.n_output - 1, -1)]
        
        if do_init:
            self.id = type(self).get_id()
            hidden_layers = self.initialize_node_genome(config)
            self.initialize_connection_genome(hidden_layers,
                                              config.init_connection_probability,
                                              config.init_connection_probability_fourier,
                                              config.weight_init_std,
                                              fourier_cutoff=-(config.num_inputs - config.n_fourier_features),
                                              force_init_path_inputs_outputs= config.force_init_path_inputs_outputs)
            
            self.update_layers()
            
            self.mutate_lr(config.mutate_sgd_lr_sigma) # initialize learning rate
        
        self.to(self.device)
 
    def generate_inputs(self, config=CPPNConfig()):
        return add_fourier_features_from_config(
            initialize_inputs_from_config(config),
            config
            )
    
    def initialize_node_genome(self, config):
            n_hidden = config.hidden_nodes_at_start
            if isinstance(n_hidden, int):
                n_hidden = (n_hidden,)
            else:
                n_hidden = tuple([int(n_hidden) for n_hidden in n_hidden])
            
            for node_id in self.input_node_ids:
                node = Node(IdentityActivation, node_id)
                self.nodes[node_id] = node
            
            for node_id in self.output_node_ids:
                if config.output_activation is None:
                    node = Node(random_choice(config.activations), node_id)
                else:
                    node = Node(config.output_activation, node_id)
                self.nodes[node_id] = node
            
            hidden_layers = {}
            for i, layer in enumerate(n_hidden):
                this_layer_id = i+1
                hidden_layers[this_layer_id] = []
                for j in range(layer):
                    new_id = type(self).get_new_node_id()
                    node = Node(random_choice(config.activations), new_id)
                    self.nodes[new_id] = node
                    node.layer = i+1
                    hidden_layers[this_layer_id].append(node)
                
            return hidden_layers
            
    
    def initialize_connection_genome(self, hidden_layers, initial_connection_prob=1.0, init_connection_prob_fourier=1.0, weight_std=1.0, fourier_cutoff=-4, force_init_path_inputs_outputs=False):
        """Initializes the connection genome of the CPPN."""
        def is_fourier(node_id):
            if init_connection_prob_fourier is None:
                return False
            return int(node_id) < fourier_cutoff
        
        prev_layer = self.input_nodes
        for layer in sorted(list(hidden_layers.values()), key=lambda x: x[0].layer if len(x) > 0 else 0):
            for node in layer:
                for prev_node in prev_layer:
                    prob = init_connection_prob_fourier if is_fourier(prev_node.id) else initial_connection_prob
                    if torch.rand(1, dtype=torch.float32) < prob:
                        self.connections[f"{prev_node.id},{node.id}"] = Connection(self.rand_weight(weight_std))
            if len(layer) > 0:
                prev_layer = layer
        
        for node in self.output_nodes:
            for prev_node in prev_layer:
                prob = init_connection_prob_fourier if is_fourier(prev_node.id) else initial_connection_prob
                if torch.rand(1, dtype=torch.float32) < prob:
                    self.connections[f"{prev_node.id},{node.id}"] = Connection(self.rand_weight(weight_std))
                    
        if force_init_path_inputs_outputs:
            for output_node in self.output_nodes:
                path = []
                path_end = output_node
                for layer in sorted(list(hidden_layers.values()), key=lambda x: x[0].layer if len(x) > 0 else 0
                                    , reverse=True):
                    if len(layer) == 0:
                        continue
                    # check to see if there is already a connection from this layer to the output node
                    if any([f"{node.id},{path_end.id}" in self.connections.keys() for node in layer]):
                        existing = [node for node in layer if f"{node.id},{path_end.id}" in self.connections.keys()][0]
                        path.append(f"{existing.id},{path_end.id}")
                        path_end = existing
                    else:
                        random_node = random_choice(layer)
                        self.connections[f"{random_node.id},{path_end.id}"] = Connection(self.rand_weight(weight_std))
                        path.append(f"{random_node.id},{path_end.id}") 
                        path_end = random_node
                    
                    
                
                if any([f"{input_node_id},{path_end.id}" in self.connections.keys() for input_node_id in self.input_node_ids]):
                    path_start = [node for node in self.input_nodes if f"{node.id},{path_end.id}" in self.connections.keys()][0]
                    path.append(f"{path_start.id},{path_end.id}")
                else:
                    random_node = random_choice(self.input_nodes)
                    self.connections[f"{random_node.id},{path_end.id}"] = Connection(self.rand_weight(weight_std))
                    path.append(f"{random_node.id},{path_end.id}")
                    
                for cx in path:
                    self.connections[cx].enabled = True
    
    def reinitialize_weights(self, config):
        for cx in self.connections.values():
            cx.weight = self.rand_weight(config.weight_init_std)
        

    def update_layers(self):
        self.enabled_connections = [conn_key for conn_key in self.connections if self.connections[conn_key].enabled]
        
        # inputs
        self.layers = [set([n.id for n in self.input_nodes])]

        # self.layers.extend(feed_forward_layers([node.id for node in self.input_nodes],
        #                                   [node.id for node in self.output_nodes],
        #                                   [(conn_key.split(',')[0], conn_key.split(',')[1]) for conn_key in self.enabled_connections]))
        self.layers.extend(feed_forward_layers(self))
        
        
        
        
        
        for layer_idx, layer in enumerate(self.layers):
            for node_id in layer:
                self.nodes[node_id].layer = layer_idx
                
        # self.node_states = {} # risky to disable, assumes node_states is reset elsewhere TODO

    def gather_inputs(self, node_id, just_w=False):
        for i in self.node_states:
            key = f"{i},{node_id}"
            if key in self.enabled_connections and key in self.connections.keys():
                if just_w:
                    yield self.connections[key].weight
                else:
                    yield self.node_states[i] * self.connections[key].weight
             
    def get_image(self, *args, **kwargs):
        self.image = self.forward(*args, **kwargs)
        return self.image

    def forward(self, x, channel_first=True, force_recalculate=True, use_graph=False, act_mode='n/a', to_cpu=False):
        # Set input node states
        for i, input_node in enumerate(self.input_nodes):
            self.node_states[input_node.id] = x[:, :, i]
        for i, output_node in enumerate(self.output_nodes):
            self.node_states[output_node.id] = torch.zeros(x.shape[0:2], device=x.device, requires_grad=False)

        # Feed forward through layers
        outputs = self.output_node_ids
        
        for layer in self.layers:
            for node_id in layer:
                # Gather inputs from incoming connections
                node_inputs = list(self.gather_inputs(node_id))
                # Sum inputs and apply activation function
                if len(node_inputs) > 0:
                    self.node_states[node_id] = self.nodes[node_id](torch.sum(torch.stack(node_inputs), dim=0))
                elif node_id not in self.node_states:
                    # TODO: shouldn't need to do this
                    self.node_states[node_id] = torch.zeros(x.shape[0:2], device=x.device, requires_grad=False)
        
        # Gather outputs
        outputs = [self.node_states[node_id] for node_id in outputs]
        outputs = torch.stack(outputs, dim=(0 if channel_first else -1))
        
        # outputs = torch.sigmoid(outputs)
        
        # normalize?
        out_range = (outputs.max() - outputs.min())
        if out_range > 0:
            outputs = (outputs - outputs.min()) / out_range
        
        # outputs = torch.nn.functional.relu(outputs)
        
        # outputs = torch.abs(outputs)
        
        outputs = torch.clamp(outputs, 0, 1)
        if to_cpu:
            return outputs.detach().cpu()
        return outputs

    def mutate(self, config:CPPNConfig, skip_update=False, pbar=False):
        """Mutates the CPPN based on the algorithm configuration."""
        add_node = config.prob_add_node
        add_connection = config.prob_add_connection
        remove_node = config.prob_remove_node
        disable_connection = config.prob_disable_connection
        mutate_weights = config.prob_mutate_weight
        mutate_bias = config.prob_mutate_bias
        mutate_activations = config.prob_mutate_activation
        mutate_sgd_lr_sigma = config.mutate_sgd_lr_sigma
        
        rng = lambda: torch.rand(1).item()
        iters = range(config.topology_mutation_iters) if not pbar else trange(config.topology_mutation_iters, leave=False)
        for _ in iters:
            if config.single_structural_mutation:
                div = max(1.0, (add_node + remove_node +
                                add_connection + disable_connection))
                r = rng()
                if r < (add_node / div):
                    self.add_node(config)
                elif r < ((add_node + remove_node) / div):
                    self.remove_node(config)
                elif r < ((add_node + remove_node +
                            add_connection) / div):
                    self.add_connection(config)
                elif r < ((add_node + remove_node +
                            add_connection + disable_connection) / div):
                    self.disable_connection()
            else:
                # mutate each structural category separately
                if rng() < add_node:
                    self.add_node(config)
                if rng() < remove_node:
                    self.remove_node(config)
                if rng() < add_connection:
                    self.add_connection(config)
                if rng() < disable_connection:
                    self.disable_connection()
        
        for _ in range(config.connection_bloat):
            self.add_connection(config)
        
        self.mutate_weights(mutate_weights, config)
        self.mutate_bias(mutate_bias, config)
        if not skip_update:
            self.update_layers()
            self.disable_invalid_connections(config)
        
        self.to(self.device) # TODO shouldn't need this
        
        self.node_states = {} # reset the node states
        
        # only mutate the learning rate and activations once per iteration
        self.mutate_activations(mutate_activations, config)
        self.mutate_lr(mutate_sgd_lr_sigma)
            
            
    
    def disable_invalid_connections(self, config):
        """Disables connections that are not compatible with the current configuration."""
        # delete any connections to or from nodes that don't exist
        for key, connection in list(self.connections.items()):
            if key.split(',')[0] not in self.nodes.keys() or key.split(',')[1] not in self.nodes.keys():
                del self.connections[key]
        
        return # TODO: test, but there should never be invalid connections
        invalid = []
        for key, connection in self.connections.items():
            if connection.enabled:
                if not is_valid_connection(self.nodes,
                                           [k.split(',') for k in self.connections.keys()],
                                           key.split(','),
                                           config,
                                           warn=True):
                    invalid.append(key)
        for key in invalid:
            key.enabled = False
            #del self.connections[key]


    def add_connection(self, config, specific_cx=None):
        """Adds a connection to the CPPN."""
        self.update_layers()
        
        for _ in range(200):  # try 200 times max
            if specific_cx is not None:
                [from_node, to_node] = specific_cx.split(',')
            else:
                [from_node, to_node] = random_choice(list(self.nodes.values()),
                                                    2, replace=False)
            if from_node.layer >= to_node.layer:
                continue  # don't allow recurrent connections
            # look to see if this connection already exists
            key = f"{from_node.id},{to_node.id}"
            if key in self.connections.keys():
                existing_cx = self.connections[key]
            else:
                existing_cx = None
            
            # if it does exist and it is disabled, there is a chance to reenable
            if existing_cx is not None:
                if not existing_cx.enabled:
                    if torch.rand(1)[0] < config.prob_reenable_connection:
                        existing_cx.enabled = True # re-enable the connection
                    break  # don't enable more than one connection
                continue # don't add more than one connection

            # else if it doesn't exist, check if it is valid
            if is_valid_connection(self.nodes,
                                           [k.split(',') for k in self.connections.keys()],
                                           key.split(','),
                                           config):
                # valid connection, add
                new_cx = Connection(self.rand_weight(config.weight_init_std, device=self.device), device=self.device)
                
                new_cx_key = f"{from_node.id},{to_node.id}"
                self.connections[new_cx_key] = new_cx
                self.update_layers()
                break # found a valid connection
            
            # else failed to find a valid connection, don't add and try again


    def add_node(self, config):
        """Adds a node to the CPPN.
            Looks for an eligible connection to split, add the node in the middle
            of the connection.
        """
        # only add nodes in the middle of non-recurrent connections (TODO)
        eligible_cxs = list(self.connections.keys())

        if len(eligible_cxs) == 0:
            return # there are no eligible connections, don't add a node

        # choose a random eligible connection
        old_cx_key = random_choice(eligible_cxs, 1, replace=False)

        # create the new node
        new_node = Node(random_choice(config.activations), type(self).get_new_node_id())
        
        assert new_node.id not in self.nodes.keys(),\
            "Node ID already exists: {}".format(new_node.id)
        
        self.nodes[new_node.id] =  new_node # add a new node between two nodes
        # self.connections[old_cx_key].enabled = False  # disable old connection
        old_weight = self.connections[old_cx_key].weight
        old_from, old_to = old_cx_key.split(',')
        del self.connections[old_cx_key]  # delete old connection

        # The connection between the first node in the chain and the
        # new node is given a weight of one and the connection between
        # the new node and the last node in the chain
        # is given the same weight as the connection being split
        
        new_cx_1_key = f"{old_from},{new_node.id}"
        new_cx_1 = Connection(torch.tensor([1.0], device=self.device))
        
        
        assert new_cx_1_key not in self.connections.keys()
        
        self.connections[new_cx_1_key] = new_cx_1

        new_cx_2_key = f"{new_node.id},{old_to}"
        new_cx_2 = Connection(old_weight)
        assert new_cx_2_key not in self.connections.keys()
        self.connections[new_cx_2_key] = new_cx_2

        self.update_layers() # update the layers of the nodes

        
    def remove_node(self, config, specific_node=None):
        """Removes a node from the CPPN.
            Only hidden nodes are eligible to be removed.
        """

        hidden = self.hidden_nodes
        
        if len(hidden) == 0 or specific_node is not None and specific_node not in hidden:
            return # no eligible nodes, don't remove a node

        # choose a random node
        if not specific_node:
            node_id_to_remove = random_choice([n.id for n in hidden], 1, False)
        else:
            node_id_to_remove = specific_node.id
        
        # delete all connections to and from the node
        for key, cx in list(self.connections.items())[::-1]:
            if node_id_to_remove in key.split(','):
                del self.connections[key]
        
        # delete the node
        for key, node in list(self.nodes.items())[::-1]:
            if key == node_id_to_remove:
                del self.nodes[key]
                break

        
        self.update_layers()
        self.disable_invalid_connections(config)


    
    def mutate_activations(self, prob, config):
        """Mutates the activation functions of the nodes."""
        if len(config.activations) == 1:
            return # no point in mutating if there is only one activation function

        eligible_nodes = self.hidden_nodes
        if config.output_activation is None:
            eligible_nodes.extend(self.output_nodes)
        if config.allow_input_activation_mutation:
            eligible_nodes.extend(self.input_nodes)
        for node in eligible_nodes:
            if torch.rand(1)[0] < prob:
                node.set_activation(random_choice(config.activations))



    def mutate_weights(self, prob, config):
        """
        Each connection weight is perturbed with a fixed probability by
        adding a floating point number chosen from a uniform distribution of
        positive and negative values """
        R_delta = torch.rand(len(self.connections.items()), device=self.device)
        R_reset = torch.rand(len(self.connections.items()), device=self.device)

        for i, connection in enumerate(self.connections.values()):
            if R_delta[i] < prob:
                delta = random_normal(None, 0, config.weight_mutation_std)
                connection.add_to_weight(delta)
                
            elif R_reset[i] < config.prob_weight_reinit:
                connection.weight = self.random_weight()

        # self.clamp_weights()


    def mutate_bias(self, prob, config):
        R_delta = torch.rand(len(self.nodes.items()), device=self.device)
        R_reset = torch.rand(len(self.nodes.items()), device=self.device)

        for i, node in enumerate(self.nodes.values()):
            if R_delta[i] < prob:
                delta = random_normal(None, 0, config.bias_mutation_std)
                node.bias = node.bias + delta
            elif R_reset[i] < config.prob_weight_reinit:
                node.bias = torch.zeros_like(node.bias)

        
    def mutate_lr(self, sigma):
        if not sigma:
            return # don't mutate
        delta =  random_normal(None, 0, sigma).item()
        self.sgd_lr = self.sgd_lr + delta
        self.sgd_lr = max(1e-8, self.sgd_lr) # prevent 0 or negative learning rates


    def prune_connections(self, config, already_pruned=0):
        if config.prune_threshold == 0 and config.min_pruned == 0:
            return 0
        removed = already_pruned
        for key, cx in list(self.connections.items())[::-1]:
            if abs(cx.weight.item()) < config.prune_threshold:
                del self.connections[key]
                removed += 1
        
        for _ in range(config.min_pruned - removed):
            if len(list(self.connections.keys())) == 0:
                return removed
            min_weight_key = min(self.connections.keys(), key=lambda k: abs(self.connections[k].weight.item()))
            removed += 1
            del self.connections[min_weight_key]
        # print("Pruned {} connections".format(removed))
        
        # TODO connections removed during node pruning not counted here
            
        return removed


    def prune_nodes(self, config):
        if config.prune_threshold_nodes == 0 and config.min_pruned_nodes == 0 and config.node_activation_prune_threshold == 0:
            return 0
        used_node_ids = []
        used_node_ids.extend(self.input_node_ids)
        used_node_ids.extend(self.output_node_ids)
        removed_nodes = 0
        removed_connections = 0
        def del_connections(node_key):
            count = 0
            for k, cx in list(self.connections.items())[::-1]:
                if node_key in k.split(','):
                    del self.connections[k]
                    count += 1
            return count
        for key in list(self.nodes.keys())[::-1]:
            incoming = list(self.gather_inputs(key, just_w=True))
            if len(incoming) == 0:
                if key not in used_node_ids:
                    self.remove_node(config, specific_node=self.nodes[key])
                    removed_nodes += 1
                continue
            
            
            l2_norm = torch.norm(torch.stack(incoming), p=2)
            
            
            if l2_norm < config.prune_threshold_nodes:
                removed_connections += del_connections(key)
                self.remove_node(config, specific_node=self.nodes[key])
                removed_nodes += 1
                continue
            
            # TODO: won't work if we clear node_states frequently
            if config.node_activation_prune_threshold > 0 and len(self.node_states)>0:
                activation = torch.abs(self.node_states.get(key, torch.tensor([0.0], device=self.device))).detach().mean()
                if activation < config.node_activation_prune_threshold:
                    removed_connections += del_connections(key)
                    self.remove_node(config, specific_node=self.nodes[key])
                    removed_nodes += 1
                    continue
        # print(f"Pruned {removed_nodes} nodes")
        return removed_nodes, removed_connections
            

    def prune(self, config):
        removed_nodes, removed_cxs = self.prune_nodes(config)
        removed_cxs += self.prune_connections(config, already_pruned=removed_cxs)
        self.update_layers()
        self.disable_invalid_connections(config)
        return removed_cxs, removed_nodes

    
    def disable_connection(self):
        """Disables a connection."""
        eligible_cxs = list(self.enabled_connections)
        if len(eligible_cxs) < 1:
            return
        cx:str = random_choice(eligible_cxs, 1, False)
        self.connections[cx].enabled = False

    def remove_connection(self):
        """Disables a connection."""
        eligible_cxs = list(self.enabled_connections)
        if len(eligible_cxs) < 1:
            return
        cx:str = random_choice(eligible_cxs, 1, False)
        del self.connections[cx]
    
    
    
    def rand_weight(self, std=1.0,device='cuda'):
        return torch.randn(1,device=device) * std
        
    def clone(self, config, cpu=False, new_id=False):
        """Clones the CPPN, optionally on the CPU. If new_id is True, the new CPPN will have a new id."""
        
        # Create the child as an empty genome
        child = CPPN(config, do_init=False)
          
        # Copy the parent's genome
        for _, node in self.nodes.items():
            child.nodes[node.id] = Node(type(node.activation), node.id, node.bias.item())
        
        for conn_key, conn in self.connections.items():
            child.connections[conn_key] = Connection(conn.weight.detach().clone())
        
        child.update_layers()
        
        # Configure record keeping information
        if new_id:
            child.id = type(self).get_id()
            child.parents = (self.id, self.id)
            child.lineage = self.lineage + [self.id]
        else:
            child.id = self.id 
            child.parents = self.parents
            child.lineage = self.lineage
            
        child.sgd_lr = self.sgd_lr
        
        if cpu:
            child.to(torch.device('cpu'))
        else:
            child.to(self.device)
            
        return child
    
    def crossover(self, other, config):
        child = self.clone(config, new_id=True)

        matching1, matching2 = get_matching_connections(
            self.connections, other.connections)
        
        # copy input and output nodes randomly
        child.nodes = nn.ModuleDict()
        
        
        for node_id in self.input_node_ids:
            node_id = str(node_id)
            from_self = np.random.rand() < .5 
            n = self.nodes[node_id] if from_self else other.nodes[node_id]
            child.nodes[node_id] = Node(n.activation, n.id, n.bias.item())
                
                
        for node_id in self.output_node_ids:
            node_id = str(node_id)
            from_self = np.random.rand() < .5 
            n = self.nodes[node_id] if from_self else other.nodes[node_id]
            child.nodes[node_id] = Node(n.activation, n.id, n.bias.item())    
        
        for match_index in range(len(matching1)):
            # Matching genes are inherited randomly
            from_self = np.random.rand() < .5 
            
            
            if from_self:
                cx_key = matching1[match_index]
                copy_cx = self.connections[cx_key]
            else:
                cx_key = matching2[match_index]
                copy_cx = other.connections[cx_key]
            
            child.connections[cx_key] = Connection(copy_cx.weight.detach().clone(), copy_cx.enabled)
            
            # Disable the connection randomly if either parent has it disabled
            self_enabled = self.connections[cx_key].enabled
            other_enabled = other.connections[cx_key].enabled
                
            if(not self_enabled or not other_enabled):
                if(np.random.rand() < 0.75):  # from Stanley/Miikulainen 2007
                    child.connections[cx_key].enabled = False
            
        
        for cx_key in child.connections.keys():
            to_node, from_node = cx_key.split(',')
            for node in [to_node, from_node]:
                if node in child.nodes.keys():
                    continue
                in_both = node in self.nodes.keys() and node in other.nodes.keys()
                if in_both:
                    from_self = np.random.rand() < .5 
                else:
                    from_self = node in self.nodes.keys()
                n = self.nodes[node] if from_self else other.nodes[node]
                child.nodes[node] = Node(n.activation, n.id, n.bias.item())
                            
        
        child.update_layers()
        child.disable_invalid_connections(config)
        
        return child

    def vis(self, x, fname='cppn_graph'):
        """Visualize the CPPN."""
        pass
        # make_dot(self.forward(x), show_attrs=True, show_saved=True, params=dict(self.named_parameters())).render(fname, format="pdf")
        
    @staticmethod
    def create_from_json(json_dict, config=None, CPPNClass=None, ConfigClass=None):
        """Constructs a CPPN from a json dict or string."""
        if isinstance(json_dict, str):
            json_dict = json.loads(json_dict, strict=False)
        if config is None and 'config' in json_dict:
            if ConfigClass is None:
                ConfigClass = CPPNConfig
            config = ConfigClass.create_from_json(json_dict["config"], ConfigClass)
            json_dict = json_dict["genome"]
            print("created config of type", type(config))
        if CPPNClass is None:
            CPPNClass = CPPN
        new_cppn = CPPNClass(config, do_init=False)
        new_cppn.from_json(json_dict)
        new_cppn.to(config.device)
        return new_cppn
    
    
    def to_json(self):
        """Converts the CPPN to a json dict."""
        return {"id":self.id,
                "parents":self.parents,
                "nodes": {k:n.to_json() for k,n in self.nodes.items()},
                "connections": {k:c.to_json() for k,c in self.connections.items()},
                "lineage": self.lineage,
                "age": self.age,
                "cell_lineage": self.cell_lineage,
                "sgd_lr": self.sgd_lr,
                "image": self.image,
                "selected": self.selected,
                }

    
    def from_json(self, json_dict):
        """Constructs a CPPN from a json dict or string."""
        if isinstance(json_dict, str):
            json_dict = json.loads(json_dict, strict=False)
        
        copy_keys = ["id", "parents", "lineage", "sgd_lr", 'age', 'cell_lineage', 'n_cells', 'selected']

        for key, value in json_dict.items():
            if key in copy_keys:
                setattr(self, key, value)

        self.nodes = nn.ModuleDict()
        self.connections = nn.ModuleDict() 
        
        for key, item in json_dict["nodes"].items():
            self.nodes[key] = Node.create_from_json(item)
        for key, item in json_dict["connections"].items():
            self.connections[key] = Connection.create_from_json(item)

        self.update_layers()
        
        
    def save(self, fname, config=None):
        """Saves the CPPN to a file."""
        import copy
        config_copy = copy.deepcopy(config)
        with open(fname, 'w') as f:
            genome_data = self.to_json()
            if config is not None:
                data = {}
                data["config"] = config_copy.to_json()
                data["genome"] = genome_data
                json.dump(data, f)
                del config_copy
            else:
                json.dump(genome_data, f)
                

    def genetic_difference(self, other) -> float:
        # only enabled connections, sorted by innovation id
        this_cxs = sorted(self.enabled_connections,
                            key=lambda c: c)
        other_cxs = sorted(other.enabled_connections,
                            key=lambda c: c)

        N = max(len(this_cxs), len(other_cxs))

        # number of excess connections
        n_excess = len(get_excess_connections(this_cxs, other_cxs))
        # number of disjoint connections
        n_disjoint = len(get_disjoint_connections(this_cxs, other_cxs))

        # matching connections
        this_matching, other_matching = get_matching_connections(
            this_cxs, other_cxs)
        
        assert len(this_matching) == len(other_matching)
        assert this_matching == other_matching
        
        difference_of_matching_weights = [
            abs(other.connections[o_cx].weight.item()-self.connections[t_cx].weight.item()) for o_cx, t_cx in zip(other_matching, this_matching)]
        # difference_of_matching_weights = torch.stack(difference_of_matching_weights)
        
        if(len(difference_of_matching_weights) == 0):
            difference_of_matching_weights = 0
        else:
            difference_of_matching_weights = sum(difference_of_matching_weights) / len(difference_of_matching_weights)

        # Furthermore, the compatibility distance function
        # includes an additional argument that counts how many
        # activation functions differ between the two individuals
        n_different_fns = 0
        for t_node, o_node in zip(self.nodes.values(), other.nodes.values()):
            if(t_node.activation.__class__.__name__ != o_node.activation.__class__.__name__):
                n_different_fns += 1

        # can normalize by size of network (from Ken's paper)
        if(N > 0):
            n_excess /= N
            n_disjoint /= N

        # weight (values from Ken)
        n_excess *= 1
        n_disjoint *= 1
        difference_of_matching_weights *= .4
        n_different_fns *= 1
        
        difference = sum([n_excess,
                            n_disjoint,
                            difference_of_matching_weights,
                            n_different_fns])
        if torch.isnan(torch.tensor(difference)):
            difference = 0

        return difference
                
# norm.py 

import logging
import os
import pandas as pd
import torch

def read_norm_data(norm_df_path, target=None):
    assert os.path.exists(norm_df_path), f"Norm data not found at {norm_df_path}"
    if norm_df_path.endswith('.csv'):
        norm_df = pd.read_csv(norm_df_path)
    elif norm_df_path.endswith('.pkl'):
        norm_df = pd.read_pickle(norm_df_path)
    else:
        raise ValueError(f"Unknown norm data format {norm_df_path}")
        
    if target is not None:
        if target in norm_df['target'].unique():
            norm_df = norm_df[norm_df['target'] == target]
        else:
            # print(f"WARNING: target {target} not found in {norm_df_path}, using mean of all targets")
            norm_df = norm_df.groupby('function').mean(numeric_only=True).reset_index()
            
    return norm_df

def read_norm_data_to_tensor(norm_df_path, target=None, run_agg='mean')->torch.Tensor:
    norm_df = read_norm_data(norm_df_path, target)
    output = torch.zeros(len(norm_df.function.unique()), 2)
    for i, fn in enumerate(norm_df.function.unique()):
        output[i,0] = norm_df[norm_df['function'] == fn][f'min_fitness_{run_agg}'].values[0]
        output[i,1] = norm_df[norm_df['function'] == fn][f'max_fitness_{run_agg}'].values[0]
    return output
    


def norm_tensor(input, norm_df, fn_name, run_agg='mean', method: str='minmax', clamp=False, warn=True):
    eps = 1e-8
    if isinstance(norm_df, str) or isinstance(norm_df, os.PathLike):
        norm_df = read_norm_data(norm_df)
    if fn_name not in norm_df['function'].unique():
        if warn:
            logging.warning(f"No norm for function {fn_name}, out of options: {norm_df['function'].unique()}")
        return input
    if method == 'minmax':
        col_name_min = f"min_fitness_{run_agg}"
        col_name_max = f"max_fitness_{run_agg}"
        if col_name_min not in norm_df.columns:
            col_name_min = "min_fitness"
        if col_name_max not in norm_df.columns:
            col_name_max = "max_fitness"
        min_ = norm_df[norm_df["function"] == fn_name][col_name_min].values[0]
        max_ = norm_df[norm_df["function"] == fn_name][col_name_max].values[0]
        
        normed = (input - min_) / (eps + max_ - min_)
        if not((normed > 0).all() and (normed < 1).all()):
            if warn:
                logging.warning(f"Normed value [{normed.min().item()}-{normed.max().item()}] out of range for function {fn_name} original was: [{input.min().item()}-{input.max().item()}]")
        
        if clamp:
            normed = torch.clamp(normed, 0, 1)
            
        return normed
    elif method == 'zscore':
        mean = norm_df[norm_df["function"] == fn_name][f"fitness_mean"].values[0]
        std = norm_df[norm_df["function"] == fn_name][f"fitness_std"].values[0]
        normed = (input - mean) / std
        if not((normed > 0).all() and (normed < 1).all()):
            logging.warning(f"Normed value [{normed.min().item()}-{normed.max().item()}] out of range for function {fn_name} original was: [{input.min().item()}-{input.max().item()}]")
        if clamp:
            normed = torch.clamp(normed, 0, 1)
        return normed
        
    else:
        raise ValueError(f"Unknown method {method}")
    
    return normed

def norm_tensor_by_tensor(input, norm):
    # not sure we should use this, since there's no guarantee that the functions are in the same order
    raise NotImplementedError
    min_ = norm[:,0][:,None]
    max_ = norm[:,1][:,None]
    normed = (input.T - min_) / (max_ - min_)
    return normed.T


def norm(df, norm, excluded_from_norm=['baseline-aggregate'], skip_conditions=[], run_agg='mean'):
    df["normed_fitness"] = df["fitness"]
    
    for fn in df["function"].unique():
        if fn in excluded_from_norm:
            continue
        this_norm = norm.loc[norm["function"] == fn]
        if len(this_norm) == 0:
            print(f"No norm for function {fn}")
            continue
        
        postfix = f"_{run_agg}" if run_agg is not None else ""
        
        if f"fitness_mean{postfix}" not in this_norm.columns:
            postfix = ""
            
        max_fitness = this_norm[f"max_fitness{postfix}"].values[0]
        min_fitness = this_norm[f"min_fitness{postfix}"].values[0]
        
        # gross:
        df.loc[(df["function"] == fn) &(~ df["condition"].isin(skip_conditions)), "normed_fitness"] = (
            (df.loc[(df["function"] == fn) &(~ df["condition"].isin(skip_conditions)), "normed_fitness"] - min_fitness) /
            (max_fitness - min_fitness))
        
        
def norm_from_saved(df, baseline_dir, excluded_from_norm=['baseline-aggregate'], skip_conditions=[],run_agg='mean'):
    baseline_norm = pd.read_pickle(os.path.join(baseline_dir, "norm.pkl"))
    norm(df, baseline_norm, excluded_from_norm, skip_conditions=skip_conditions, run_agg=run_agg)    
    
    
# sgd_weights_clip.py
import matplotlib.pyplot as plt
import logging
import os
import torch
from torchvision.transforms import Resize


from tqdm import trange
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience:int=1, min_delta:float=0, n_genomes=0, device='cpu'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = torch.inf
        
        self.counter_tensor = torch.zeros(n_genomes, device=device)
        self.min_loss_tensor = torch.ones(n_genomes, device=device)*torch.inf

    def check_stop(self, loss:float) -> bool:
        if loss < (self.min_loss + self.min_delta):
            self.min_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def mask_stop(self, loss):
        loss=loss.flatten()
        improve = loss < (self.min_loss_tensor + self.min_delta)
        
        self.counter_tensor = torch.where(improve, 0, self.counter_tensor+1)
        self.min_loss_tensor = torch.where(improve, loss, self.min_loss_tensor)
        
        result = self.counter_tensor >= self.patience

        # shorten tensors
        # self.min_loss_tensor = torch.where(result, torch.nan, self.min_loss_tensor) # replace the losses that are stopped with nan
        # self.min_loss_tensor = self.min_loss_tensor[~torch.isnan(self.min_loss_tensor)] # remove nan
        # self.counter_tensor = torch.where(result, torch.nan, self.counter_tensor) # replace the counters that are stopped with nan
        # self.counter_tensor = self.counter_tensor[~torch.isnan(self.counter_tensor)] # remove nan
        
        return result

resize = Resize((32, 32),antialias=True)
def min_resize(imgs):
    if imgs.shape[-1] < 32 or imgs.shape[-2] < 32:
        return resize(imgs)
    return imgs

def anneal_steps_sigmoid(gen, max_gens):
    # https://www.desmos.com/calculator/x8mntiv4bc
    a = 200 # max steps
    c = 0.01 # steepness
    x = torch.tensor(gen)
    d = int(max_gens*0.5) # halfway point of annealing
    steps = int(torch.floor(a / (1+torch.exp(-c*(x-d)))-1).item())
    steps = max(steps, 0)
    return steps

def anneal_steps_linear(gen, max_gens):
    # https://www.desmos.com/calculator/30cpcsfwzm
    a = 800 # max steps
    c = 4 # steepness
    d = 5 # delay
    x = torch.tensor(gen) if not isinstance(gen, torch.Tensor) else gen
    steps = int(torch.floor(c*(x-d)).item())
    steps = max(steps, 0)
    steps = min(steps, a)
    return steps
    

def save_annealing_plot(max_gens, fn, config):
    X = torch.linspace(0, max_gens, max_gens)
    Y = [fn(x, max_gens) for x in X]
    plt.plot(X, Y)
    plt.xlabel("Generation")
    plt.ylabel("SGD Steps")
    plt.title("SGD Steps by Generation")
    plt.savefig(os.path.join(config.run_output_dir, "sgd_steps_annealing.pdf"))
    plt.close()


def anneal(config, sgd_steps, genomes, current_gen):
    anneal_fn = None
    
    if isinstance(sgd_steps, str) and sgd_steps == 'annealing-sigmoid':
        anneal_fn = anneal_steps_sigmoid
    elif isinstance(sgd_steps, str) and sgd_steps == 'annealing-linear':
        anneal_fn = anneal_steps_linear
        
    # do annealing:
    sgd_steps = anneal_fn(current_gen, config.num_generations)
    
    print("Annealing steps:", sgd_steps)
    if current_gen == 0:
        save_annealing_plot(config.num_generations, anneal_fn, config)
    if sgd_steps == 0:
        # apply a small random mutation to the weights
        for c in genomes:
            for p in c.parameters():
                if isinstance(p, torch.nn.Parameter):
                    p.data += torch.randn_like(p.data) * config.sgd_learning_rate
                else:
                    p += torch.randn(1).item() * config.sgd_learning_rate

def batch_lr_mod(inputs, config, all_params, lr):
    # Modify the learning rate by the batch size
    # Linear Scaling Rule: When the minibatch size is
    # multiplied by k, multiply the learning rate by k.
    # https://arxiv.org/pdf/1706.02677.pdf
    k = len(inputs)
    mod = config.batch_lr_mod * k
    lr = lr * mod
    for param_group in all_params:
        param_group['lr'] = param_group['lr'] * mod


def prep_images(imgs, config):
    assert torch.isfinite(imgs).all(), "NaNs in images"
    
    if len(config.color_mode) == 1:
        imgs = imgs.repeat(1, 3, 1, 1) # grayscale to RGB
    
    imgs = min_resize(imgs)
    
    return imgs


def sgd_weights(genomes, inputs, target, fns, config, norm=None, record_loss=None, skip_pbar=False, current_gen=0):
    lr = config.sgd_learning_rate
    sgd_steps = config.sgd_steps
    early_stop = config.sgd_early_stop
    
    if isinstance(sgd_steps, str) and 'annealing' in sgd_steps:
       anneal(config, sgd_steps, genomes, current_gen)
                        
    
    # if isinstance(genomes[0], tuple):
        # genomes = [g for c_,ci_,g in genomes]

    # if mask is not None:
        # filter fns to only the ones that are enabled in mask
        # fns = [fn for i, fn in enumerate(fns) if mask[i].any()]
        # mask = mask[mask.any(dim=1)]
    
        
    all_params = []
    for c in genomes:
        if True:
            c.sgd_lr = lr # TODO CONFIG
        all_params.extend([{'params': list(c.parameters()), 'lr': c.sgd_lr}])
        for p in c.parameters():
            p.requires_grad = True

    if len(fns) == 0 or len(all_params) == 0:
        print("No fitness functions or no parameters to optimize. Skipping.")
        return 0 # took no steps
    
    if config.batch_lr_mod:
        batch_lr_mod(inputs, config, all_params, lr)
    
    avg_lr = sum([p['lr'] for p in all_params])/len(all_params)
    
    # All CPPN weights in one optimizer 
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=config.sgd_l2_reg)
    
    # Compile function
    def f(X, *gs):
        return torch.stack([g(X, force_recalculate=True, use_graph=True, channel_first=True) for g in gs[0]])
    
    if not skip_pbar:
        pbar = trange(sgd_steps, leave=False, disable=sgd_steps <= 5)

    # loss function
    def loss_fn(imgs, target, return_all=False):
        # prepare images
        imgs = prep_images(imgs, config)
 
        # calculate fitness
        normed = torch.zeros((imgs.shape[0], len(fns)), device=imgs.device)
        for i, fn in enumerate(fns):
            fitness = fn(imgs, target)

            if norm is not None:
                normed_fit = norm_tensor(fitness, norm, fn.__name__, clamp=True, warn=False)
            else:
                normed_fit = fitness # no normalization
            normed[:, i] = normed_fit
                
        # if mask is not None:
            # normed = normed * mask.T # mask loss by the functions in each cell
            
        assert torch.isfinite(normed).all()

        # loss is the inverse of the mean fitness
        inv = torch.sub(1.0, normed)
        if return_all:
            return inv.mean(), inv.mean(dim=1)
        return inv.mean()
        
    
    n_params_total = sum([len(list(c.parameters())) for c in genomes])
    
    # Optimize
    fwd_passes = 0
    step = 0
    stopping = EarlyStopping(patience=early_stop if early_stop else sgd_steps, min_delta=config.sgd_early_stop_delta, n_genomes=len(genomes), device=config.device)
    stop_mask = torch.zeros(len(genomes), dtype=torch.bool, device=config.device)
    if skip_pbar:
        pbar = range(sgd_steps)
    
    # this_genomes = genomes
    # this_mask = mask
    
    for step in pbar:
        this_target = target[:torch.sum(~stop_mask)] # take only the non stopped targets
        this_genomes =  [n for i,(n) in enumerate(genomes) if not stop_mask[i]]
        # if mask is not None:
            # this_mask = mask[:, ~stop_mask]
            # stop_mask = stop_mask[~stop_mask]
            # mask = mask * ~stop_mask

        for i,(c) in enumerate(genomes):
            for p in c.parameters():
                # freeze the parameters of the stopped genomes
                if c not in this_genomes:
                    p.requires_grad = False # manually freeze the parameters (after optimizer is defined, this is okay)
                # else:
                    # p.requires_grad = True
        n_params = sum([len(list(c.parameters())) for i,c in enumerate(this_genomes)])
        
        imgs = f(inputs, this_genomes)
        loss, all_loss = loss_fn(imgs, this_target, True)
        
        # fill in torch.inf for the stopped genomes
        for i,(c) in enumerate(genomes):
            if stop_mask[i]:
                all_loss = torch.cat((all_loss[:i], torch.tensor([torch.inf], device=config.device), all_loss[i:]))
        
        if record_loss is not None:
            record_loss[step] = loss.item()
        
        if not loss.requires_grad:
            print("No gradients")
            return step # no gradients, so we're done
        
        assert torch.isfinite(loss).all(), "Non-finite loss"
        
        optimizer.zero_grad()
        
        try:
            loss.backward() 
        except RuntimeError as e:
            logging.warning("RuntimeError in loss.backward()")
            import traceback
            traceback.print_exc()
            return step
        
        # make nan grads 0
        # TODO: prevent this upstream
        for param_group in all_params:
            for param in param_group['params']:
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        logging.warning(f"Non-finite gradients: {param.grad}")
                    param.grad[param.grad != param.grad] = 0       
                else:
                    param.grad = torch.zeros_like(param)
        
        if config.sgd_clamp_grad:
            torch.nn.utils.clip_grad_norm_(all_params, config.sgd_clamp_grad, error_if_nonfinite=True)
        
        if len(all_params) > 0:
            optimizer.step()
            
        if config.max_weight:
            for p in all_params:
                for param in p['params']:
                    param.data.clamp_(-config.max_weight, config.max_weight)
        
        fwd_passes += len(this_genomes)
        
        if early_stop:
            stop_mask = stopping.mask_stop(all_loss)
            if torch.sum(~stop_mask) == 0:
                break
        
        for param_group in all_params:
            for param in param_group['params']:
                assert torch.isfinite(param).all(), "Non-finite parameters after step"

        # if early_stop and stopping.check_stop(loss.item()):
            # break
        
        
        if isinstance(pbar, tqdm):
            pbar.set_postfix_str(f"loss={loss.detach().clone().mean().item():.3f}")
            pbar.set_description_str(f"Optimizing {n_params}/{n_params_total} params on {len(this_genomes)}/{len(genomes)} genomes and {len(fns)} fns lr: {avg_lr:.2e}")
    
    return step+1


"""
Require: Distribution over tasks P (T ), outer step size η, regularization strength λ,
2: while not converged do
3: Sample mini-batch of tasks {Ti}B
i=1 ∼ P (T )
4: for Each task Ti do
5: Compute task meta-gradient gi = Implicit-Meta-Gradient(Ti, θ, λ)
6: end for
7: Average above gradients to get ˆ∇F (θ) = (1/B) ∑B
i=1 gi
8: Update meta-parameters with gradient descent: θ ← θ − η ˆ∇F (θ) // (or Adam)
9: end while
"""
def sgd_weights_imaml(genomes, mask, inputs, target, fns, norm, config, early_stop=3, record_loss=None, skip_pbar=False, current_gen=0):
    assert mask is not None, "IMAML requires a cell-function mask"
    
    param_deltas = []
    total_steps = 0
    losses = []
    
    for i,task in enumerate(fns):
        these_genomes = [(None,None,genome.clone(config, new_id=False)) for _,_,genome in genomes]
        params_before = [torch.tensor(list(c.parameters()), device=config.device) for _,_,c in these_genomes]

        print("Task", i, task.__name__)
    
        total_steps += sgd_weights(these_genomes,
                                   None,
                                   inputs,
                                   target,
                                   [task],
                                   norm,
                                   config,
                                   early_stop,
                                   record_loss,
                                   skip_pbar,
                                   current_gen
                                   )
        if record_loss is not None:
            losses.append(record_loss.clone())
        
        params_after = [torch.tensor(list(c.parameters()), device=config.device) for _,_,c in these_genomes]
        
        delta = [p1-p0 for p0,p1 in zip(params_before, params_after)]
        param_deltas.append(delta)
    
    if record_loss is not None:
        record_loss[:] = torch.stack(losses).mean(dim=0) # record the average loss by step
    
    delta_mag = 0 # for recording

    if True:
        
        total_index = 0
        # apply the average delta to the genomes
        for i in range(len(genomes)):               # for each child genome
            for j,cell_mask in enumerate(mask.T):   # for each cell
                avg_delta = torch.stack([d[i]*cell_mask[fn_index] for fn_index,d in enumerate(param_deltas)])
                avg_delta = avg_delta.mean(dim=0)   # average the deltas for the cell
                delta_mag += torch.norm(avg_delta)  # record      
                
                if total_index < len(genomes):
                    genomes[total_index] = (total_index,None,genomes[i][-1].clone(config, new_id=False))
                else:
                    genomes.append((total_index,None,genomes[i][-1].clone(config, new_id=False)))
                
                for p,d in zip(genomes[total_index][-1].parameters(), avg_delta):
                    p.data += d
                total_index += 1
                    
    else:
        # apply the average delta to the genomes
        for i in range(len(genomes)):               # for each child genome
            avg_delta = torch.stack([d[i] for d in param_deltas])
            avg_delta = avg_delta.mean(dim=0)
            delta_mag += torch.norm(avg_delta)
            for p,d in zip(genomes[i][-1].parameters(), avg_delta):
                p.data += d
                
                
    print("Ended up with", len(genomes), "genomes")
    print("\t Avg. parameter delta magnitude:", delta_mag.item()/len(genomes))
    
    return total_steps
        
    
    
    
    
    
    
    
# clip-utils.py


import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

import clip



CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_NORM = torchvision.transforms.Normalize(CLIP_MEAN, CLIP_STD)  # normalize an image that is already scaled to [0, 1]
CLIP_RESIZE = torchvision.transforms.Resize((224, 224))

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model_vit, _ = clip.load("ViT-B/32", device=device, jit=False)
clip_model_rn, _ = clip.load("RN50", device=device, jit=False)
clip_model_vit.eval()
clip_model_rn.eval()


@torch.no_grad()
def embed_text(text: str):
    assert isinstance(text, str)
    text = clip.tokenize(text).to(device)
    text_features_vit = clip_model_vit.encode_text(text)
    text_features_rn = clip_model_rn.encode_text(text)
    return torch.cat([text_features_vit, text_features_rn], dim=-1) # [N, 1024]


def embed_images(images):
    images = CLIP_NORM(images)
    images = CLIP_RESIZE(images)
    image_features_vit = clip_model_vit.encode_image(images)  # [N, 512]
    image_features_rn = clip_model_rn.encode_image(images)  # [N, 512]
    emb = torch.cat([image_features_vit, image_features_rn], dim=-1) # [N, 1024]
    return emb


cached_text_features = {}

variance_weight = 0.0 # TODO config


def fit_fn(imgs, target):
    if target in cached_text_features:
        text_features = cached_text_features[target]
    else:
        text_features = embed_text(target)
        cached_text_features[target] = text_features
    image_features = embed_images(imgs)
    clip_sim = torch.cosine_similarity(text_features, image_features, dim=-1)
    
    var = imgs.var(dim=(1, 2, 3))
    fitness = (1.0-variance_weight)*clip_sim + (variance_weight)*var
    return fitness

def sgd(population, target, conf):
    X = population[0].generate_inputs(conf)
    record_loss = np.ones(conf.sgd_steps) * np.nan
    n_steps = sgd_weights(population, X, target, [fit_fn], conf, record_loss=record_loss)
    return record_loss[:n_steps] 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__== "__main__":
    size = (4, 4)
    
    # coordinates
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    inputs = initialize_inputs(size[0], size[1],
                               True, False,
                               3, device,
                               coord_range=(-1,1))
    
    
    # config
    config = CPPNConfig()
    config.color_mode = 'L'
    config.num_outputs = 1
    config.sgd_steps = 1000
    config.sgd_early_stop = 100
    config.sgd_learning_rate = 0.1
    config.hidden_nodes_at_start = [0]
    config.activations = [nn.Tanh()]
    
    # cppn
    cppn = CPPN(config).to(device)  
      
    print(f"Number of parameters: {get_n_params(cppn)}")
    
    # forward pass
    output = cppn(inputs,channel_first=False)
    print(output.shape)
    
    
    
    target = "white"
    
    # test the sgd
    n_steps = sgd_weights([cppn], inputs, target, [fit_fn], config, record_loss=None, skip_pbar=False, current_gen=0)
    print(n_steps)
    
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TKAgg')
    plt.imshow(output.detach().cpu().numpy(), cmap='gray')
    plt.show()
    plt.savefig(f'test.png')

  
