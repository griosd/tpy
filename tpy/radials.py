import torch
from torch.autograd import Function
from .core import TpModule, numpy
from scipy import special


class TorchGammainc(Function):
    @staticmethod
    def forward(ctx, a, x):
        ctx.save_for_backward(a, x)
        return torch.as_tensor(special.gammainc(numpy(a), numpy(x)), device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output, eps = 1e-4):
        a,x = ctx.saved_tensors
        grad_a = grad_x = None
        if ctx.needs_input_grad[0]: #replace a by grad_output
            grad_a = (TorchGammainc.apply(grad_output*(1+eps), x)-TorchGammainc.apply(grad_output*(1-eps), x))/(2*grad_output*eps)
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


class TorchBetainc(Function):
    @staticmethod
    def forward(ctx, a, b, x):
        ctx.save_for_backward(a, b, x)
        return torch.as_tensor(special.betainc(numpy(a), numpy(b), numpy(x)), device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output, eps = 1e-4):
        a, b, x = ctx.saved_tensors
        grad_a = grad_b = grad_x = None
        if ctx.needs_input_grad[0]: #replace a by grad_output
            grad_a = (TorchBetainc.apply(grad_output*(1+eps), b, x)-TorchBetainc.apply(grad_output*(1-eps), b, x))/(2*grad_output*eps)
        if ctx.needs_input_grad[1]:  # replace b by grad_output
            grad_b = (TorchBetainc.apply(a, grad_output * (1 + eps), x) - TorchBetainc.apply(a, grad_output * (1 - eps), x)) / (2 * grad_output * eps)
        if ctx.needs_input_grad[2]: #replace x by grad_output
            grad_x = (torch.lgamma(a + b) - torch.lgamma(a) - torch.lgamma(b)).exp() * torch.pow(grad_output, a-1) * torch.pow(1-grad_output, b-1)
        return grad_a, grad_b, grad_x
betainc = TorchBetainc.apply


class TorchBetaincinv(Function):
    @staticmethod
    def forward(ctx, a, b, y):
        ctx.save_for_backward(a, b, y)
        return torch.as_tensor(special.betaincinv(numpy(a), numpy(b), numpy(y)), device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        a, b, y = ctx.saved_tensors
        grad_a = grad_b = grad_y = None
        if ctx.needs_input_grad[0]: #replace a by grad_output
            grad_a = 1 / TorchBetainc.backward(ctx, TorchBetaincinv.apply(grad_output, b, y))[0]
        if ctx.needs_input_grad[1]: #replace b by grad_output
            grad_b = 1 / TorchBetainc.backward(ctx, TorchBetaincinv.apply(a, grad_output, y))[1]
        if ctx.needs_input_grad[2]: #replace y by grad_output
            grad_y = 1 / TorchBetainc.backward(ctx, TorchBetaincinv.apply(a, b, grad_output))[2]
        return grad_a, grad_b, grad_y
betaincinv = TorchBetaincinv.apply


class CDF(TpModule):
    def __init__(self, *args, **kwargs):
        super(CDF, self).__init__(*args, **kwargs)

    def F(self, x, n=None):
        pass

    def Q(self, y, n=None):
        pass


class NormGaussian(CDF):
    '''2-Norm of a Gaussian based on Chi'''
    def __init__(self, *args, **kwargs):
        super(NormGaussian, self).__init__(*args, **kwargs)

    def F(self, x, n=1):
        return gammainc(n/2, x**2/2)

    def Q(self, y, n=1):
        return (2*gammaincinv(n/2, y))**0.5


class NormStudentT(CDF):
    '''2-Norm of a Student-t based on FisherSnedecor (scaled sqrt)'''
    def __init__(self, v=3, *args, **kwargs):
        super(NormStudentT, self).__init__(*args, **kwargs)
        self.v = v

    def F(self, x, n=1):
        x1 = (x ** 2) * (self.v) / (n * (self.v  - 2))
        dx1 = (n * x1) / (n * x1 + self.v )
        return betainc(n / 2, (self.v ) / 2, dx1)

    def Q(self, y, n=1):
        dx1 = betaincinv(n / 2, (self.v) / 2, y)
        x1 = dx1 / (n - dx1 * n)
        return (x1 * n * (self.v - 2)) ** 0.5


class GammaInverseSqrt(CDF):
    def __init__(self, v=3, *args, **kwargs):
        ''' $\sqrt{\Gamma^{-1}}$ to produce Student-t from Gaussian'''
        super(GammaInverseSqrt, self).__init__(*args, **kwargs)
        self.v = v

    def F(self, x, n=None):
        beta = (self.v - 2) / 2
        return 1-gammainc(self.v/ 2, beta / (x ** 2))

    def Q(self, y, n=None):
        beta = (self.v - 2) / 2
        return (beta / gammaincinv(self.v  / 2, 1-y)) ** 0.5


class Weibull(CDF):
    def __init__(self, l=1, k=1, *args, **kwargs):
        ''' Weibull'''
        super(Weibull, self).__init__(*args, **kwargs)
        self.l = l
        self.k = k

    def F(self, x, n=None):
        return 1 - torch.exp(-(x/self.l)**self.k)

    def Q(self, y, n=None):
        return self.l*(-torch.log(1-y))**(1/self.k)


class Uniform(CDF):
    def __init__(self, l=0, r=1, *args, **kwargs):
        ''' Uniform'''
        super(Uniform, self).__init__(*args, **kwargs)
        self.l = l
        self.r = r

    def F(self, x, n=None):
        return (x - self.l) / (self.r - self.l)

    def Q(self, y, n=None):
        return y * (self.r - self.l) + self.l


class Pareto(CDF):
    def __init__(self, s=0, a=1, *args, **kwargs):
        ''' Uniform'''
        super(Pareto, self).__init__(*args, **kwargs)
        self.s = s
        self.a = a

    def F(self, x, n=None):
        return 1 - (self.s/x)**self.a

    def Q(self, y, n=None):
        return self.s/((1 -y)**(1/self.a))
