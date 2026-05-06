"""Utility functions for argparse in training/generation scripts."""


def str2bool(v: str) -> bool:
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError(f'Boolean value expected, got {v}')


def str2dtype(v: str):
    """Convert string to torch dtype for argparse."""
    import torch
    dtypes = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
        'int8': torch.int8,
    }
    if v.lower() in dtypes:
        return dtypes[v.lower()]
    raise ValueError(f'Unsupported dtype: {v}')
