import matplotlib.pyplot as plt
import logging
import os
import torch
from torchvision.transforms import Resize

import __main__ as main

# if not hasattr(main, '__file__'):
#     try:
#         from tqdm.notebook import trange
#     except ImportError:
#         from tqdm import trange
# else:
from tqdm import trange
from tqdm import tqdm
from cppn.norm import norm_tensor, norm_tensor_by_tensor
# from cppn.torch_genome import TorchFeedForwardNetwork


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
        self.min_loss_tensor = torch.where(result, torch.nan, self.min_loss_tensor) # replace the losses that are stopped with nan
        self.min_loss_tensor = self.min_loss_tensor[~torch.isnan(self.min_loss_tensor)] # remove nan
        self.counter_tensor = torch.where(result, torch.nan, self.counter_tensor) # replace the counters that are stopped with nan
        self.counter_tensor = self.counter_tensor[~torch.isnan(self.counter_tensor)] # remove nan
        
        return result
        
        if loss < (self.min_loss + self.min_delta):
            self.min_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


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



def sgd_weights(genomes, inputs, target, config, fns, norm=None, record_loss=None, skip_pbar=False,  mask=None, current_gen=0):
    sgd_steps = config.sgd_steps
    lr = config.sgd_learning_rate
   
    anneal_fn = None
    if isinstance(sgd_steps, str) and sgd_steps == 'annealing-sigmoid':
        anneal_fn = anneal_steps_sigmoid
    elif isinstance(sgd_steps, str) and sgd_steps == 'annealing-linear':
        anneal_fn = anneal_steps_linear
        
    if isinstance(sgd_steps, str) and 'annealing' in sgd_steps:
        sgd_steps = anneal_fn(current_gen, config.num_generations)
        print("Annealing steps:", sgd_steps)
        if current_gen == 0:
            save_annealing_plot(config.num_generations, anneal_fn, config)
        if sgd_steps == 0:
            # apply a small random mutation to the weights
            for c in genomes:
                for p in c.parameters:
                    if isinstance(p, torch.nn.Parameter):
                        p.data += torch.randn_like(p.data) * lr
                    else:
                        p += torch.randn(1).item() * lr
            

    early_stop = config.sgd_early_stop

    if isinstance(genomes[0], tuple):
        genomes = [g for c_,ci_,g in genomes]
    all_params = []

    if mask is not None:
        # filter fns to only the ones that are enabled in mask
        fns = [fn for i, fn in enumerate(fns) if mask[i].any()]
        mask = mask[mask.any(dim=1)]
    
    if len(fns) == 0:
        print("No fitness functions")
        return 0
        
    for c in genomes:
        c.prep_params()
        # c.loss_delta = 0.0
        # c.last_loss = 0.0
    
        P = list(c.parameters)
        for p in P:
            p.requires_grad = True
            # print(p)

        # print(c.key)
        this_lr = c.sgd_lr if hasattr(c, 'sgd_lr') else config.sgd_learning_rate
        # all_params.extend([{'params': P, 'lr': this_lr}])
        all_params.extend(P)
    
    if len(all_params) == 0:
        print("No parameters to optimize")
        return 0 # took no steps
    
    
    # All CPPN weights in one optimizer 

    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=config.sgd_l2_reg)
    # optimizer = torch.optim.SGD(all_params, lr=lr, weight_decay=config.sgd_l2_reg)


    # Objective function
    def f(X, *gs):
        # return torch.stack([g(X, force_recalculate=True, use_graph=True, channel_first=True) for g in gs[0]])
        outputs = torch.stack([g.activate(X) for g in gs])
        return outputs
    
    compiled_fn = f
    if not skip_pbar:
        pbar = trange(sgd_steps, leave=False, disable=sgd_steps <= 5)
    
    if len(target.shape) == 3:
        target = target.expand(len(genomes), -1, -1, -1)
    else:
        target = target.expand(len(genomes), -1, -1)
    
    # loss function
    def loss_fn(imgs, target, return_all=False):
        # prepare images
        # remove nan
        assert torch.isfinite(imgs).all(), "NaNs in images"
        
        
        # if len(config.color_mode) == 1:
            # imgs = imgs.repeat(1, 3, 1, 1) # grayscale to RGB
        
        imgs = min_resize(imgs)
 
        # calculate fitness
        pop_size = imgs.shape[0]
        normed = torch.zeros((pop_size, len(fns)), device=imgs.device)
        for i, fn in enumerate(fns):
            fitness = fn(imgs, target)

            if norm is not None:
                normed_fit = norm_tensor(fitness, norm, fn.__name__, clamp=True, warn=False)
            else:
                normed_fit = fitness # no normalization
            normed[:, i] = normed_fit
        
        # for i, c in enumerate(genomes):
            # c.loss_delta += normed[i].mean().item() - c.last_loss
            # c.last_loss = normed[i].mean().item()
                
        if mask is not None:
            # mask loss by the functions in each cell
            normed = normed * mask.T
            
        assert torch.isfinite(normed).all()

        # return the inverse of the mean fitness
        inv = torch.sub(1.0, normed)
        if return_all:
            return inv.mean(), inv
        return inv.mean()
    
    n_params_total = sum([len(list(c.parameters)) for c in genomes])

    
    # get networks
    # nets = [
    #     TorchFeedForwardNetwork.create(genome, neat_config, inputs.shape[1:], config.device) for genome in genomes
    # ]
    
    # for net in nets:
    #     for key in net.values:
    #         net.values[key] = torch.zeros_like(inputs[0])
    
    # Optimize
    step = 0
    fwd_passes = 0
    stopping = EarlyStopping(patience=early_stop if early_stop else sgd_steps, min_delta=config.sgd_early_stop_delta, n_genomes=len(genomes), device=config.device)
    stop_mask = torch.zeros(len(genomes), dtype=torch.bool)
    if skip_pbar:
        pbar = range(sgd_steps)
        
    # this_nets = nets
    this_genomes = genomes
    for step in pbar:
        
        this_nets = [n for i,n in enumerate(this_nets) if not stop_mask[i]]
        this_target = target[:torch.sum(~stop_mask)] # take only the non stopped targets
        this_genomes =  [n for i,n in enumerate(this_genomes) if not stop_mask[i]]
        for i,c in enumerate(genomes):
            for p in c.parameters:
                # freeze the parameters of the stopped genomes
                if c not in this_genomes:
                    p.requires_grad = False # manually freeze the parameters (after optimizer is defined, this is okay)
                
        n_params = sum([len(list(c.parameters)) for i,c in enumerate(this_genomes)])
        
        imgs = compiled_fn(inputs, *this_nets)
        loss, all_loss = loss_fn(imgs, this_target, True)
        
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
        
        if config.sgd_clamp_grad:
            torch.nn.utils.clip_grad_norm_(all_params, config.sgd_clamp_grad, error_if_nonfinite=True)
        
        
        if len(all_params) > 0:
            optimizer.step()
        
        if config.max_weight:
            for p in all_params:
                p.data.clamp_(-config.max_weight, config.max_weight)
        
        if early_stop:
            stop_mask = stopping.mask_stop(all_loss)
            if torch.sum(~stop_mask) == 0:
                break
        
        # if early_stop and stopping.check_stop(loss.item()):
        #     break
        
        fwd_passes += len(this_nets)
        
        if isinstance(pbar, tqdm):
            pbar.set_postfix_str(f"loss={loss.detach().clone().mean().item():.3f}")
            # pbar.set_description_str(f"Optimizing {n_params} params on {len(genomes)} genomes and {len(fns)} fns lr: {lr:.2e} avg:{avg_lr:.2e}")
            pbar.set_description_str(f"Optimizing {n_params}/{n_params_total} params on {len(this_genomes)}/{len(genomes)} genomes and {len(fns)} fns lr: {lr:.2e}")
    
    
    for p in all_params:
        p.requires_grad = False
    
    print(f"W min/mean/max {torch.cat([p.view(-1) for p in all_params]).min().item():0.3f}, {torch.cat([p.view(-1) for p in all_params]).mean().item():0.3f}, {torch.cat([p.view(-1) for p in all_params]).max().item():0.3f}")
    
    return step+1, fwd_passes
        