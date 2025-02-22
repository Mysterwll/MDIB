import numpy as np
import torch
import random

def seed_everything(seed: int) -> None:
    import monai

    monai.utils.set_determinism(seed=seed, additional_settings=None)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False