import copy
import sys
from itertools import count
from random import choice, random, shuffle
# from neat.genome import DefaultGenomeConfig, DefaultNodeGene, DefaultConnectionGene, ConfigParameter
# from torch_attributes import FloatAttribute, BoolAttribute, StringAttribute
# from neat.genes import BaseGene
# from neat.config import ConfigParameter, DefaultClassConfig
# from neat.genome import DefaultGenome

# from neat.activations import ActivationFunctionSet
# from neat.aggregations import AggregationFunctionSet
# from neat.config import ConfigParameter, write_pretty_params
# from neat.genes import DefaultConnectionGene, DefaultNodeGene
# from neat.graphs import creates_cycle
# from neat.graphs import required_for_output

from neat.six_util import itervalues

from cppn import CPPN, CPPNConfig


# class TorchNodeGene(BaseGene):
#     _gene_attributes = [FloatAttribute('bias'),
#                         FloatAttribute('response'),
#                         StringAttribute('activation', options=''),
#                         StringAttribute('aggregation', options='')]

#     def __init__(self, key):
#         assert isinstance(key, int), f"DefaultNodeGene key must be an int, not {key!r}"
#         BaseGene.__init__(self, key)

#     def distance(self, other, config):
#         d = abs(self.bias - other.bias) + abs(self.response - other.response)
#         if self.activation != other.activation:
#             d += 1.0
#         if self.aggregation != other.aggregation:
#             d += 1.0
#         return d * config.compatibility_weight_coefficient
    
#     @classmethod
#     def validate_attributes(cls, config):
#         for a in cls._gene_attributes:
#             a.validate(config)

# class TorchConnectionGene(BaseGene):
#     _gene_attributes = [FloatAttribute('weight'),
#                         BoolAttribute('enabled')]

#     def __init__(self, key):
#         assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(key)
#         BaseGene.__init__(self, key)

#     def distance(self, other, config):
#         d = abs(self.weight - other.weight)
#         if self.enabled != other.enabled:
#             d += 1.0
#         return d * config.compatibility_weight_coefficient   
#     @classmethod
#     def validate_attributes(cls, config):
#         for a in cls._gene_attributes:
#             a.validate(config)

    

# class CPPNGenomeConfig(DefaultGenomeConfig):
#     def __init__(self, params):
#         params['cppn_config'] = CPPNConfig()
#         self.cppn_config = params['cppn_config']
#         # return super().__init__(params)
        
#         # Create full set of available activation functions.
#         self.activation_defs = ActivationFunctionSet()
#         # ditto for aggregation functions - name difference for backward compatibility
#         self.aggregation_function_defs = AggregationFunctionSet()
#         self.aggregation_defs = self.aggregation_function_defs

#         self._params = [ConfigParameter('num_inputs', int),
#                         ConfigParameter('num_outputs', int),
#                         ConfigParameter('num_hidden', int),
#                         ConfigParameter('feed_forward', bool),
#                         ConfigParameter('compatibility_disjoint_coefficient', float),
#                         ConfigParameter('compatibility_weight_coefficient', float),
#                         ConfigParameter('conn_add_prob', float),
#                         ConfigParameter('conn_delete_prob', float),
#                         ConfigParameter('node_add_prob', float),
#                         ConfigParameter('node_delete_prob', float),
#                         ConfigParameter('single_structural_mutation', bool, 'false'),
#                         ConfigParameter('structural_mutation_surer', str, 'default'),
#                         ConfigParameter('initial_connection', str, 'unconnected')]
        
#         self._params += [ConfigParameter('max_connections', int, -1),
#                         ConfigParameter('max_nodes', int, -1)]
        
#         # Gather configuration data from the gene classes.
#         self.node_gene_type = params['node_gene_type']
#         self._params += self.node_gene_type.get_config_params()
#         self.connection_gene_type = params['connection_gene_type']
#         self._params += self.connection_gene_type.get_config_params()

#         # Use the configuration data to interpret the supplied parameters.
#         for p in self._params:
#             setattr(self, p.name, p.interpret(params))

#         self.node_gene_type.validate_attributes(self)
#         self.connection_gene_type.validate_attributes(self)

#         # By convention, input pins have negative keys, and the output
#         # pins have keys 0,1,...
#         self.input_keys = [-i - 1 for i in range(self.num_inputs)]
#         self.output_keys = [i for i in range(self.num_outputs)]

#         self.connection_fraction = None

#         # Verify that initial connection type is valid.
#         # pylint: disable=access-member-before-definition
#         if 'partial' in self.initial_connection:
#             c, p = self.initial_connection.split()
#             self.initial_connection = c
#             self.connection_fraction = float(p)
#             if not (0 <= self.connection_fraction <= 1):
#                 raise RuntimeError(
#                     "'partial' connection value must be between 0.0 and 1.0, inclusive.")

#         assert self.initial_connection in self.allowed_connectivity

#         # Verify structural_mutation_surer is valid.
#         # pylint: disable=access-member-before-definition
#         if self.structural_mutation_surer.lower() in ['1', 'yes', 'true', 'on']:
#             self.structural_mutation_surer = 'true'
#         elif self.structural_mutation_surer.lower() in ['0', 'no', 'false', 'off']:
#             self.structural_mutation_surer = 'false'
#         elif self.structural_mutation_surer.lower() == 'default':
#             self.structural_mutation_surer = 'default'
#         else:
#             error_string = f"Invalid structural_mutation_surer {self.structural_mutation_surer!r}"
#             raise RuntimeError(error_string)

#         self.node_indexer = None
        
#         self.genome_config = self
        


# class TorchGenome(DefaultGenome):
#     """
#     A genome for generalized neural networks.

#     Terminology
#         pin: Point at which the network is conceptually connected to the external world;
#              pins are either input or output.
#         node: Analog of a physical neuron.
#         connection: Connection between a pin/node output and a node's input, or between a node's
#              output and a pin/node input.
#         key: Identifier for an object, unique within the set of similar objects.

#     Design assumptions and conventions.
#         1. Each output pin is connected only to the output of its own unique
#            neuron by an implicit connection with weight one. This connection
#            is permanently enabled.
#         2. The output pin's key is always the same as the key for its
#            associated neuron.
#         3. Output neurons can be modified but not deleted.
#         4. The input values are applied to the input pins unmodified.
#     """

#     # @property
#     # def connections(self):
#     #     return self.cppn.connections
    
#     @property
#     def parameters(self):
#         params = []
#         for cg in self.connections.values():
#             if cg.enabled:
#                 # if not isinstance(cg.weight, torch.Tensor):
#                     # necessary because add_node adds weight of 1.0 (TODO)
#                     # cg.weight = torch.tensor(cg.weight).float() 
#                 # w = torch.nn.Parameter(cg.weight.detach())
#                 # w.requires_grad = True
#                 params.append(cg.weight)
#                 # cg.weight = w
#         for ng in self.nodes.values():
#             # b = torch.nn.Parameter(ng.bias.detach())
#             # r = torch.nn.Parameter(ng.response.detach())
#             # b.requires_grad = True
#             # r.requires_grad = True
#             params.append(ng.bias)
#             params.append(ng.response)
#             # ng.bias = b
#             # ng.response = r
#             ...
#         return params
    
#     def prep_params(self):
#         for cg in self.connections.values():
#             if cg.enabled:
#                 if not isinstance(cg.weight, torch.Tensor):
#                     cg.weight = torch.tensor(cg.weight).float()
#                 w = torch.nn.Parameter(cg.weight.detach())
#                 w.requires_grad = True
#                 cg.weight = w

#         for ng in self.nodes.values():
#             b = torch.nn.Parameter(ng.bias.detach())
#             r = torch.nn.Parameter(ng.response.detach())
#             b.requires_grad = True
#             r.requires_grad = True
#             ng.bias = b
#             ng.response = r
    
#     def bloat(self, rate, config, as_absolute=False):
#         # add num_cx * rate new connections
        
#         num_cx = len(self.connections)

#         if as_absolute:
#            new_cx = int(rate)
#         else:
#             new_cx = int(num_cx * rate) 
        
#         for _ in range(new_cx):
#             done = False
#             tries = 0
#             while not done and tries < 10:
#                 tries += 1
#                 cx_before = len(self.connections)
#                 self.mutate_add_connection(config.genome_config)
#                 done = len(self.connections) > cx_before
                
        
#         num_added = len(self.connections) - num_cx
#         return num_added
    
#     def prune(self, num):
#         # remove num connections with the lowest weight
#         if num > len(self.connections):
#             num = len(self.connections)
        
#         cx_keys_sorted = sorted(self.connections.keys(), key=lambda k: abs(self.connections[k].weight))
#         for k in cx_keys_sorted[:num]:
#             del self.connections[k]
#         return num
    
#     def prune_threshold(self, threshold):
#         # remove connections with weight < threshold
#         cx_keys = list(self.connections.keys())
#         for k in cx_keys:
#             if abs(self.connections[k].weight) < threshold:
#                 del self.connections[k]
#         return len(cx_keys) - len(self.connections)
    
#     def mutate_add_node(self, config):
#         # enforce a max
#         if config.genome_config.max_nodes>=0 and len(self.nodes) >= config.genome_config.max_nodes:
#             return 
#         super().mutate_add_node(config)
    
#     def mutate_add_connection(self, config):
#         # enforce a max
#         if config.genome_config.max_connections>=0 and len(self.connections) >= config.genome_config.max_connections:
#             return
#         super().mutate_add_connection(config)
    
#     @classmethod
#     def parse_config(cls, param_dict):
#         param_dict['node_gene_type'] = TorchNodeGene
#         param_dict['connection_gene_type'] = TorchConnectionGene
        
#         return CPPNGenomeConfig(param_dict)
#     # ,
#     #                             [ConfigParameter('max_nodes', int, -1),
#     #                              ConfigParameter('max_connections', int, -1),])
                                

    

# from neat.config import ConfigParameter
# import warnings
# import torch
# import traceback

# def interpret(self, config_dict):
#         """
#         Converts the config_parser output into the proper type,
#         supplies defaults if available and needed, and checks for some errors.
#         """
#         value = config_dict.get(self.name)
#         if value is None:
#             if self.default is None:
#                 raise RuntimeError('Missing configuration item: ' + self.name)
#             else:
#                 warnings.warn("Using default {!r} for '{!s}'".format(self.default, self.name),
#                               DeprecationWarning)
#                 if (str != self.value_type) and isinstance(self.default, self.value_type):
#                     return self.default
#                 else:
#                     value = self.default

#         try:
#             if str == self.value_type:
#                 return str(value)
#             if int == self.value_type:
#                 return int(value)
#             if bool == self.value_type:
#                 if value.lower() == "true":
#                     return True
#                 elif value.lower() == "false":
#                     return False
#                 else:
#                     raise RuntimeError(self.name + " must be True or False")
#             if float == self.value_type:
#                 return float(value)
#             if torch.Tensor == self.value_type:
#                 return torch.tensor(float(value)).float()
#             if list == self.value_type:
#                 return value.split(" ")
#         except Exception:
#             raise RuntimeError("Error interpreting config item '{}' with value {!r} and type {}".format(
#                 self.name, value, self.value_type))
            

#         raise RuntimeError("Unexpected configuration type: " + repr(self.value_type))


# from neat.graphs import feed_forward_layers


class TorchFeedForwardNetwork(object):
    def __init__(self, inputs, outputs, node_evals, shape=(1, 1), device="cpu"):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, torch.zeros(shape, device=device)) for key in inputs + outputs)

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = []
            for i, w in links:
                if i in self.values:
                    node_inputs.append(self.values[i] * w)
                else:
                    node_inputs.append(torch.zeros_like(inputs[0]) * w)
            s = agg_func(node_inputs)
            self.values[node] = act_func(bias + response * s)

        outputs = torch.stack([self.values[i] for i in self.output_nodes])
        outputs = 1.0 - torch.abs(outputs)
        # outputs = torch.sigmoid(5.0*outputs)
        return outputs

    def to(self, device):
        for k in self.values:
            self.values[k] = self.values[k].to(device)
        for i in range(len(self.node_evals)):
            node, act_func, agg_func, bias, response, links = self.node_evals[i]
            self.node_evals[i] = (node, act_func, agg_func, bias, response, [(i, torch.tensor(w, device=device)) for i, w in links])
        return self

    @staticmethod
    def create(genome, config, shape=(1, 1), device="cpu"):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in itervalues(genome.connections) if cg.enabled]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                # node_expr = [] # currently unused
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight))
                        # node_expr.append("v[{}] * {:.7e}".format(inode, cg.weight))


                ng = genome.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)
                node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

        return TorchFeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals, shape, device)



# ConfigParameter.interpret = interpret


#TODO:



def feed_forward_layers(inputs, outputs, connections):
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

    required = required_for_output(inputs, outputs, connections)

    layers = []
    s = set(inputs)
    
    dangling_inputs = set()
    for n in required:
        has_input = False
        has_output = False
        for (a, b) in connections:
            if b == n:
                has_input = True
                break
            if a==n:
                has_output = True
                
        if has_output and not has_input:
            dangling_inputs.add(n)
            
    # add dangling inputs to the input set
    s = s.union(dangling_inputs)
    
    while 1:

        c = get_candidate_nodes(s, connections)
        t = set()
        for n in c:
            # check if all inputs are in s
            all_inputs_in_s = all(a in s for (a, b) in connections if b == n)
            if n in required and all_inputs_in_s:
                t.add(n)
        # t = set(a for (a, b) in connections if b in s and a not in s)
        if not t:
            break

        layers.append(t)
        s = s.union(t)
    return layers


def get_candidate_nodes(s, connections):
    """Find candidate nodes c for the next layer.  These nodes should connect
    a node in s to a node not in s."""
    return set(b for (a, b) in connections if a in s and b not in s)


def creates_cycle(connections, test):
    """
    Returns true if the addition of the 'test' connection would create a cycle,
    assuming that no cycle already exists in the graph represented by 'connections'.
    """
    i, o = test
    if i == o:
        return True

    visited = {o}
    while True:
        num_added = 0
        for a, b in connections:
            if a in visited and b not in visited:
                if b == i:
                    return True

                visited.add(b)
                num_added += 1

        if num_added == 0:
            return False


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

    required = set(outputs) # outputs always required
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