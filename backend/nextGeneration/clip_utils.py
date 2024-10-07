import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

import clip

from nextGeneration.cppn.sgd_weights_clip import sgd_weights


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