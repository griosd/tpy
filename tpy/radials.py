import torch
from torch.autograd import Function
from .core import TpModule, numpy
from scipy import special


class TorchGammainc(Function):
    @staticmethod
    def forward(ctx, a, x):
        ctx.save_for_backward(a,x)
        return torch.as_tensor(special.gammainc(numpy(a), numpy(x)), device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output, eps = 1e-4):
        a,x = ctx.saved_tensors
        grad_a = grad_x = None
        if ctx.needs_input_grad[0]: #replace a by grad_output
            grad_a = (TorchGammainc.apply(grad_output*(1+eps),x)-TorchGammainc.apply(grad_output*(1-eps),x))/(2*grad_output*eps)
        if ctx.needs_input_grad[1]: #replace x by grad_output
            grad_x = torch.pow(grad_output, a-1) * torch.exp(-grad_output) / torch.lgamma(a).exp()
        return grad_a, grad_x
gammainc = TorchGammainc.apply


class TorchGammaincinv(Function):
    @staticmethod
    def forward(ctx, a, y):
        ctx.save_for_backward(a, y)
        return torch.as_tensor(special.gammaincinv(numpy(a), numpy(y)), device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        a, y = ctx.saved_tensors
        grad_a = grad_y = None
        if ctx.needs_input_grad[0]: #replace a by grad_output
            grad_a = 1 / TorchGammainc.backward(ctx, TorchGammaincinv.apply(grad_output, y))[0]
        if ctx.needs_input_grad[1]: #replace y by grad_output
            grad_y = 1 / TorchGammainc.backward(ctx, TorchGammaincinv.apply(a, grad_output))[1]
        return grad_a, grad_y
gammaincinv = TorchGammaincinv.apply


class CDF(TpModule):
    def __init__(self, *args, **kwargs):
        super(CDF, self).__init__(*args, **kwargs)

    def F(self, x, n=None):
        pass

    def Q(self, y, n=None):
        pass


class Chi(CDF):
    def __init__(self, *args, **kwargs):
        super(Chi, self).__init__(*args, **kwargs)

    def F(self, x, n=1):
        return gammainc(n/2, x**2/2)

    def Q(self, y, n=None):
        return (2*gammaincinv(n/2, y))**0.5


class GammaInverse(CDF):
    def __init__(self, v=3, *args, **kwargs):
        super(GammaInverse, self).__init__(*args, **kwargs)
        self.v = v

    def F(self, x):
        beta = (self.v - 2) / 2
        return 1-gammainc(self.v / 2, beta / (x ** 2))

    def Q(self, y):
        beta = (self.v - 2) / 2
        return (beta / gammaincinv(self.v / 2, 1-y)) ** 0.5



def resolve_inverse(function, value):
    pass