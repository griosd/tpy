import torch
from .core import TpModule, device_fn


class Deterministic(TpModule):
    def forward(self, t):
        pass


class Poly(Deterministic):
    def __init__(self, n=1, w=None, *args, **kwargs):
        super(Poly, self).__init__(*args, **kwargs)
        if w is None:
            w = torch.ones(n+1, device=self.device).view(-1, n+1, 1)
        self.w = w
        self.e = torch.arange(self.w.shape[1], device=self.device, dtype=torch.float32).view(1, -1, 1).contiguous()

    def forward(self, t):
        return torch.mul(self.w, t**self.e).sum(dim=1)


class Zero(Deterministic):
    def __init__(self,*args, **kwargs):
        super(Zero, self).__init__(*args, **kwargs)

    def forward(self, t):
        return torch.zeros_like(t)


class Warped(TpModule):
    def __init__(self, *args, **kwargs):
        super(Warped, self).__init__(*args, **kwargs)

    def forward(self, t, x):
        pass

    def inverse(self, t, y):
        pass

    def gradient_inverse(self, t, y, eps=1e-4):
        return (self.inverse(t, y*(1+eps)+eps)-self.inverse(t, y*(1-eps)-eps))/(2*eps*(y+1))


class Affine(Warped):
    def __init__(self, shift=None, scale=None, pol=0, *args, **kwargs):
        super(Affine, self).__init__(*args, **kwargs)
        if shift is None:
            shift = Poly(n=pol)
        if scale is None:
            scale = Poly(n=pol)
        self.shift = shift
        self.scale = scale

    def forward(self, t, x):
        return self.shift(t)[:, :, None] + x * self.scale(t)[:, :, None]

    def inverse(self, t, y):
        return (y - self.shift(t)[:, :, None]) / self.scale(t)[:, :, None]

    def gradient_inverse(self, t, y):
        return 1/self.scale(t)[:, :, None]
