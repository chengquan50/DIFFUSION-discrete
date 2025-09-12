import torch
import numpy as np
import random

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x/steps)+s)/(1+s)*np.pi*0.5)**2
    alphas_cumprod = alphas_cumprod/alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:]/alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def gumbel_softmax_sample(logits, temperature=1.0, hard=False):
    eps = 1e-20
    U = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
    y = (logits + gumbel_noise) / temperature
    return torch.nn.functional.softmax(y, dim=-1) if not hard else torch.nn.functional.gumbel_softmax(logits, tau=temperature, hard=True)

def extract(arr, t, x_shape):
    batch_size = t.shape[0]
    out = arr[t].reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return out

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
