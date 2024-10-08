"""Contains the CPPN, Node, and Connection classes."""
import json
from nextGeneration.cppn.util import *
from nextGeneration.cppn.graph_util import *
import nextGeneration.cppn.activation_functions as af
from nextGeneration.cppn.config import CPPNConfig
from nextGeneration.cppn.fourier_features import add_fourier_features_from_config

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
            json["activation"] = af.ACTIVATION_FUNCTIONS[json["activation"]]
        self.id = json["id"]
        self.set_activation(af.__dict__[json["activation"]])
        self.bias = nn.Parameter(torch.tensor(json["bias"]))
        self.layer = json["layer"]
    
    @staticmethod
    def create_from_json(json):
        if isinstance(json["activation"], str):
            json["activation"] = af.ACTIVATION_FUNCTIONS[json["activation"]]
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
                node = Node(af.IdentityActivation, node_id)
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

        self.layers.extend(feed_forward_layers([node.id for node in self.input_nodes],
                                          [node.id for node in self.output_nodes],
                                          [(conn_key.split(',')[0], conn_key.split(',')[1]) for conn_key in self.enabled_connections]))
        
        
        
        
        
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
        make_dot(self.forward(x), show_attrs=True, show_saved=True, params=dict(self.named_parameters())).render(fname, format="pdf")
        
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
                
    
if __name__== "__main__":
    from cppn.fourier_features import add_fourier_features
    from torchviz import make_dot
    
    size = (256, 256)
    
    # coordinates
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    inputs = initialize_inputs(size[0], size[1],
                               True, False,
                               3, device,
                               coord_range=(-1,1))
    
    inputs = add_fourier_features(
            inputs,
            4,
            .65,
            dims=2,
            include_original=True,
            )
    
    # cppn
    cppn = CPPN(inputs.shape[-1], 3, (32,16), .99).to(device)    
    print(f"Number of parameters: {get_n_params(cppn)}")
    
    # forward pass
    output = cppn(inputs)
    
    import imageio.v2 as imageio
    import cv2
    
    
    target = imageio.imread('../data/sunrise.png', pilmode='RGB')
    # resize
    target = cv2.resize(target, size) / 255.0
    target = torch.tensor(target, dtype=torch.float32, device=device)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(cppn.parameters(), lr=1e-3)
    
    from tqdm import trange
    pbar = trange(100000)
    images = []
    try:
        for step in pbar:
            output = cppn(inputs)
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.4f}")
            
            if step % 1 == 0:
                images.append(output.detach().cpu().numpy())
            
    except KeyboardInterrupt:
        pass  
    
      
    import matplotlib.pyplot as plt
    plt.imshow(output.detach().cpu().numpy(), cmap='gray')
    plt.savefig(f'test.png')

    # make gif
    import numpy as np
    imageio.mimsave('test.gif', [np.array(img) for img in images], fps=60)

    make_dot(output, params=dict(cppn.named_parameters())).render("attached", format="png")
