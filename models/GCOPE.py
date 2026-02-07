import torch.nn as nn

class Pretrain(nn.Module):
    """Placeholder module. Pretraining is not supported for this model."""
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Invalid usage: pretraining is not supported via this interface."
        )


class Downstream(nn.Module):
    """Placeholder module. Downstream execution is not supported for this model."""
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Invalid usage: downstream execution is not supported via this interface."
        )
