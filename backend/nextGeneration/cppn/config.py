"""Stores configuration parameters for the CPPN."""
import inspect
import json
from typing import Callable
import imageio.v2 as iio
import torch
import logging

from nextGeneration.cppn.activation_functions import *
from nextGeneration.cppn.name_to_fn import name_to_fn

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