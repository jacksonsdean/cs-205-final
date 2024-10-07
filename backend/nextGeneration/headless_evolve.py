from cppn import *
from cppn.sgd_weights_clip import *
# from cppn.util import *
import matplotlib.pyplot as plt
import numpy as np
from cppn.util import *
from cppn.fourier_features import *

if __name__ == "__main__":

    conf = CPPNConfig()
    conf.device = "cuda"
    conf.n_fourier_features = 0
    conf.num_inputs = 2 + conf.n_fourier_features + int(conf.use_input_bias) + int(conf.use_radial_distance)
    conf.sgd_steps = 500
    conf.sgd_early_stop = 10
    conf.sgd_learning_rate = 10.0

    conf.connection_bloat = 1

    variance_weight = 0.25


    X = add_fourier_features_from_config(initialize_inputs_from_config(conf),
                                        conf)

    import clip_utils

    target = "a circle"
    text_features = clip_utils.embed_text(target)

    cached_text_features = {
        target:
        text_features
        }

    def fn(imgs, target):
        if target in cached_text_features:
            text_features = cached_text_features[target]
        else:
            text_features = clip_utils.embed_text(target)
            cached_text_features[target] = text_features
        image_features = clip_utils.embed_images(imgs)
        clip_sim = torch.cosine_similarity(text_features, image_features, dim=-1)
        
        var = imgs.var(dim=(1, 2, 3))
        fitness = (1.0-variance_weight)*clip_sim + (variance_weight)*var
        return fitness

    pop_size = 20
    n_gens = 2000
    survival_rate = 0.25

    highest_fit = 0.0
    best_genome = None

    fits_over_time = np.ones((n_gens*conf.sgd_steps)) * np.nan
    population = [(CPPN(conf), 0.0) for _ in range(pop_size)]

    all_losses = []
    try:
        for gen in range(n_gens):
            population = sorted(population, key=lambda x: x[1], reverse=True)
            if population[0][1] > highest_fit:
                highest_fit = population[0][1]
                best_genome = population[0][0]
                print(f"Generation {gen}: {highest_fit}")
            n_survivors = int(pop_size * survival_rate)
            population = population[:n_survivors]
            # clone and mutate the top 3 candidates and add them to the population
            for i in range(pop_size-n_survivors):
                new_cand = population[i%n_survivors][0].clone(conf)
                new_cand.mutate(conf)
                population.append((new_cand, 0.0))
                

            record_loss = np.ones(conf.sgd_steps) * np.nan
            candidates = [x[0] for x in population]
            try:
                n_steps = sgd_weights(candidates, X, target, [fn], conf, record_loss=record_loss)
            except KeyboardInterrupt:
                raise
            all_losses.append(record_loss)
            
            fs = torch.stack([g[0](X, force_recalculate=True, use_graph=True, channel_first=True) for g in population])
            for i, (cand, _) in enumerate(population):
                population[i] = (cand, fs[i].mean().item())
            fits_over_time[gen*conf.sgd_steps] = fs.mean().item()
    except KeyboardInterrupt:
        pass
            


    all_losses = np.array(all_losses).reshape(-1)

    xs = np.arange(len(all_losses))
    series = all_losses.astype(np.double)
    mask = np.isfinite(series)
    plt.plot(xs[mask], series[mask], linestyle='-', label='losses')
    ax = plt.twinx()
    xs = np.arange(len(fits_over_time))
    series = fits_over_time.astype(np.double)
    mask = np.isfinite(series)
    ax.plot(xs[mask], series[mask], linestyle='-', marker='o', label='fits', color='green')

    # plt.plot(fits_over_time, color="red", label="Fits", marker="o")
    plt.xlabel("SGD step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"output/losses_{target}.png")
    # plt.show()
    plt.close()

    plt.imshow(best_genome.get_image(X, channel_first=False, to_cpu=True))
    plt.savefig(f"output/best_{target}.png")
    # plt.show()
    plt.close()
