export const API_URL = process.env.REACT_APP_NODE_ENV === "local" ? "http://127.1.1.1:5000" : (process.env.REACT_APP_NODE_ENV === "production" ? "https://5c9gzxp0r0.execute-api.us-east-1.amazonaws.com/prod/next-gen" : "https://vln9z6a7gi.execute-api.us-east-1.amazonaws.com/development/next-gen")

export const DEFAULT_CONFIG = {
    "color_mode": "RGB",
    "res_w": 128,
    "res_h": 128,
    "population_size": 20,
    "activations": ["IdentityActivation", "GaussActivation", "SinActivation", "CosActivation", "TanhActivation", "SigmoidActivation"],
    "output_activation": "",
    "use_radial_distance": true,
    "prob_mutate_weight": 0.35,
    "prob_add_connection": 0.4,
    "prob_disable_connection": 0.25,
    "prob_mutate_activation": 0.1,
    "prob_add_node": 0.4,
    "prob_remove_node": 0.4,
    "prob_crossover": .25,
    "version":  [2, 0, 0],
    "fourier_seed": 0,
    "sgd_learning_rate": 1.0,
    "sgd_steps": 10,
}

export const RESET_OPERATION = "reset"
export const NEXT_GEN_OPERATION = "next_gen"
export const SAVE_IMAGES_OPERATION = "save_images"

export const POST_FORMAT = {
    "operation": RESET_OPERATION, "config": DEFAULT_CONFIG, "population": [], "clip_text": ""
}

export const MAX_HISTORY = 20