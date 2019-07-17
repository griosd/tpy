import torch
from .core import TpModule, device_fn


class Deterministic(TpModule):
    def forward(self, t):
        pass

    def __mul__(self, other):
        return DeterministicProd(self, other)

    def __add__(self, other):
        return DeterministicSum(self, other)

    def __matmul__(self, other):
        return DeterministicComp(self, other)


class DeterministicOperations(Deterministic):
    def __init__(self,  _f1: Deterministic, _f2: Deterministic, *args, **kwargs):
        super(DeterministicOperations, self).__init__(*args, **kwargs)
        self.f1 = _f1
        self.f2 = _f2


class DeterministicProd(DeterministicOperations):
    def forward(self, t):
        return self.f1(t) * self.f2(t)


class DeterministicSum(DeterministicOperations):
    def forward(self, t):
        return self.f1(t) + self.f2(t)


class DeterministicComp(DeterministicOperations):
    def forward(self, t):
        return self.f1(self.f2(t))


class Func(Deterministic):
    def __init__(self, f, *args, **kwargs):
        super(Func, self).__init__(*args, **kwargs)
        self.f = f

    def forward(self, t):
        return self.f(t)


class Zero(Deterministic):
    def __init__(self,*args, **kwargs):
        super(Zero, self).__init__(*args, **kwargs)

    def forward(self, t):
        return torch.zeros_like(t).view(1, -1, 1)


class Constant(Deterministic):
    def __init__(self, c=None, *args, **kwargs):
        super(Constant, self).__init__(*args, **kwargs)
        if c is None:
            c = torch.ones(1, device=self.device).view(-1, 1, 1)
        self.c = c

    def forward(self, t):
        return self.c.repeat((1, t.shape[0], 1))


class Poly(Deterministic):
    def __init__(self, n=1, w=None, *args, **kwargs):
        super(Poly, self).__init__(*args, **kwargs)
        if w is None:
            w = torch.ones(n+1, device=self.device).view(-1, n+1, 1)
        self.w = w
        self.e = torch.arange(self.w.shape[1], device=self.device, dtype=torch.float32).view(1, -1, 1).contiguous()

    def forward(self, t):
        return torch.mul(self.w, t**self.e).sum(dim=1).view(-1, t.shape[0], 1)


class Marginal(TpModule):
    def __init__(self, *args, **kwargs):
        super(Marginal, self).__init__(*args, **kwargs)

    def forward(self, t, x):
        pass

    def inverse(self, t, y):
        pass

    def log_gradient_inverse(self, t, y):
        eps = 1e-4
        return torch.log((self.inverse(t, y*(1+eps)+eps)-self.inverse(t, y*(1-eps)-eps))/(2*eps*(y+1)))


class Affine(Marginal):
    def __init__(self, shift=None, scale=None, pol=0, *args, **kwargs):
        super(Affine, self).__init__(*args, **kwargs)
        if shift is None:
            shift = Poly(n=pol)
        if scale is None:
            scale = Poly(n=pol)
        self.shift = shift
        self.scale = scale

    def forward(self, t, x):
        return self.shift(t) + x * self.scale(t)

    def inverse(self, t, y):
        return (y - self.shift(t)) / self.scale(t)

    def log_gradient_inverse(self, t, y):
        return -self.scale(t).log()


class Shift(Marginal):
    def __init__(self, shift=None, pol=0, *args, **kwargs):
        super(Shift, self).__init__(*args, **kwargs)
        if shift is None:
            shift = Poly(n=pol)
        self.shift = shift

    def forward(self, t, x):
        return self.shift(t) + x

    def inverse(self, t, y):
        return y - self.shift(t)

    def log_gradient_inverse(self, t, y):
        return torch.zeros_like(y)


class LogShift(Marginal):
    def __init__(self, shift=None, *args, **kwargs):
        super(LogShift, self).__init__(*args, **kwargs)
        if shift is None:
            shift = Constant()
        self.shift = shift

    def forward(self, t, x):
        return self.shift(t) + torch.exp(x)

    def inverse(self, t, y):
        return torch.log(y - self.shift(t))

    def log_gradient_inverse(self, t, y):
        return -(y - self.shift(t)).log()


class BoxCoxShift(Marginal):
    def __init__(self, power=None, shift=None, *args, **kwargs):
        super(BoxCoxShift, self).__init__(*args, **kwargs)
        if shift is None:
            shift = Constant()
        if power is None:
            power = Constant()
        self.shift = shift
        self.power = power

    def forward(self, t, x):
        power = self.power(t)
        scaled = (power*x) + 1.0
        transformed = torch.sign(scaled) * (torch.abs(scaled) ** (1.0 / power))
        return self.shift(t)+transformed

    def inverse(self, t, y):
        shifted = y - self.shift(t)
        power = self.power(t)
        return ((torch.sign(shifted) * torch.abs(shifted) ** power)-1.0)/power

    def log_gradient_inverse(self, t, y):
        shifted = y - self.shift(t)
        power = self.power(t)
        return (power-1)*torch.max(torch.abs(shifted), y*0+1e-6).log()
