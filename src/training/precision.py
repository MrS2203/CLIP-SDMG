import torch
from contextlib import suppress


def get_autocast(precision):
    if precision == 'amp':
        return lambda: torch.amp.autocast(device_type='cuda')
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    else:
        return suppress
