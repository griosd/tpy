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
    def backward(ctx, grad_output, eps=1e-4):
        a, x = ctx.saved_tensors
        grad_a = grad_x = None
        if ctx.needs_input_grad[0]:  # replace a by grad_output
            grad_a = grad_output*(TorchGammainc.apply(a*(1+eps), x)-TorchGammainc.apply(a*(1-eps), x))/(2*a*eps)
        if ctx.needs_input_grad[1]:  # replace x by grad_output
            grad_x = grad_output*((a-1)*torch.log(x) - x - torch.lgamma(a)).exp()
        return grad_a, grad_x


gammainc = TorchGammainc.apply


class TorchGammaincinv(Function):
    @staticmethod
    def forward(ctx, a, y):
        ctx.save_for_backward(a, y)
        return torch.as_tensor(special.gammaincinv(numpy(a), numpy(y)), device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output, eps=1e-4):
        a, y = ctx.saved_tensors
        grad_a = grad_y = None
        if ctx.needs_input_grad[0]:  # replace a by grad_output
            # grad_a = 1 / TorchGammainc.backward(ctx, TorchGammaincinv.apply(a, y))[0]
            grad_a = grad_output * (TorchGammaincinv.apply(a * (1+eps), y) - TorchGammaincinv.apply(a * (1-eps),
                                                                                                    y)) / (2*a*eps)
        if ctx.needs_input_grad[1]:  # replace y by grad_output
            # grad_y = 1 / TorchGammainc.backward(ctx, TorchGammaincinv.apply(a, y))[1]
            grad_y = grad_output*(TorchGammaincinv.apply(a, y*(1+eps)) - TorchGammaincinv.apply(a,
                                                                                                y*(1-eps))) / (2*y*eps)
        return grad_a, grad_y


gammaincinv = TorchGammaincinv.apply


class TorchBetainc(Function):
    @staticmethod
    def forward(ctx, a, b, x):
        ctx.save_for_backward(a, b, x)
        return torch.as_tensor(special.betainc(numpy(a), numpy(b), numpy(x)), device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output, eps=1e-4):
        a, b, x = ctx.saved_tensors
        grad_a = grad_b = grad_x = None
        if ctx.needs_input_grad[0]:  # replace a by grad_output
            grad_a = grad_output*(TorchBetainc.apply(a*(1+eps), b, x)-TorchBetainc.apply(a*(1-eps), b, x))/(2*a*eps)
        if ctx.needs_input_grad[1]:  # replace b by grad_output
            grad_b = grad_output*(TorchBetainc.apply(a, b*(1+eps), x) - TorchBetainc.apply(a, b*(1-eps), x)) / (2*b*eps)
        if ctx.needs_input_grad[2]:  # replace x by grad_output
            grad_x = grad_output*(torch.lgamma(a+b)-torch.lgamma(a)-torch.lgamma(b)).exp() * \
                     torch.pow(x, a-1)*torch.pow(1-x, b-1)
        return grad_a, grad_b, grad_x


betainc = TorchBetainc.apply


class TorchBetaincinv(Function):
    @staticmethod
    def forward(ctx, a, b, y):
        ctx.save_for_backward(a, b, y)
        return torch.as_tensor(special.betaincinv(numpy(a), numpy(b), numpy(y)), device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output, eps=1e-4):
        a, b, y = ctx.saved_tensors
        grad_a = grad_b = grad_y = None
        if ctx.needs_input_grad[0]:  # replace a by grad_output
            grad_a = grad_output*(TorchBetaincinv.apply(a*(1+eps), b, y) - TorchBetaincinv.apply(a*(1-eps),
                                                                                                 b, y))/(2*a*eps)
        if ctx.needs_input_grad[1]:  # replace b by grad_output
            grad_b = grad_output*(TorchBetaincinv.apply(a, b*(1+eps), y) - TorchBetaincinv.apply(a, b*(1-eps),
                                                                                                 y))/(2*b*eps)
        if ctx.needs_input_grad[2]:  # replace y by grad_output
            grad_y = grad_output*(TorchBetaincinv.apply(a, b, y*(1+eps)) - TorchBetaincinv.apply(a, b,
                                                                                                 y*(1-eps)))/(2*y*eps)
        return grad_a, grad_b, grad_y


betaincinv = TorchBetaincinv.apply


class CDF(TpModule):
    def __init__(self, *args, **kwargs):
        super(CDF, self).__init__(*args, **kwargs)

    def F(self, x, n=None):
        pass

    def Q(self, y, n=None):
        pass


class TransfNorm(CDF):
    def __init__(self, norm, trans, *args, **kwargs):
        super(TransfNorm, self).__init__(*args, **kwargs)
        self.norm = norm
        self.trans = trans

    def F(self, x, n=1):
        return self.trans(self.norm.F(x, n))

    def Q(self, y, n=1):
        return self.norm.Q(self.trans(y), n)


def FisherSnedecorF(x, n, v):
    dx1 = (n * x) / (n * x + v)
    y = betainc(n/2, v/2, dx1)
    return y


def FisherSnedecorQ(y, n, v):
    dx1 = betaincinv(n/2, v/2, y)
    x = v * dx1 / (n-n*dx1)
    return x


class NormGaussian(CDF):
    """2-Norm of a Gaussian based on Chi"""
    def __init__(self, *args, **kwargs):
        super(NormGaussian, self).__init__(*args, **kwargs)

    def F(self, x, n=1):
        n = torch.as_tensor(n, dtype=x.dtype, device=x.device)
        return gammainc(n/2, x**2/2)

    def Q(self, y, n=1):
        n = torch.as_tensor(n, dtype=y.dtype, device=y.device)
        return (2*gammaincinv(n/2, y))**0.5


class NormStudentT(CDF):
    """2-Norm of a Student-t based on FisherSnedecor (scaled sqrt)"""
    def __init__(self, v=3, *args, **kwargs):
        super(NormStudentT, self).__init__(*args, **kwargs)
        self.v = v

    @property
    def _v(self):
        return self.v + 2.0

    def F(self, x, n=1):
        n = torch.as_tensor(n, dtype=x.dtype, device=x.device)
        return FisherSnedecorF((x ** 2) / n, n, self._v)

    def Q(self, y, n=1):
        n = torch.as_tensor(n, dtype=y.dtype, device=y.device)
        return (n * FisherSnedecorQ(y, n, self._v)) ** 0.5

    def posterior_F(self, x, n=1, n_obs=0, norm_y_obs=0):
        n, n_obs = torch.as_tensor([n, n_obs], dtype=x.dtype, device=x.device)
        factor = (self._v + n_obs)/(self._v + norm_y_obs ** 2)
        return FisherSnedecorF((x ** 2) * factor / n, n, self._v+n_obs)

    def posterior_Q(self, y, n=1, n_obs=0, norm_y_obs=0):
        n, n_obs = torch.as_tensor([n, n_obs], dtype=y.dtype, device=y.device)
        # n = torch.as_tensor(n, dtype=y.dtype, device=y.device)
        factor_inv = (self._v + norm_y_obs ** 2) / (self._v + n_obs)
        return (n * FisherSnedecorQ(y, n, self._v+n_obs) * factor_inv) ** 0.5


class GammaInverseSqrt(CDF):
    def __init__(self, v=3, *args, **kwargs):
        """ $sqrt{Gamma^{-1}}$ to produce Student-t from Gaussian"""
        super(GammaInverseSqrt, self).__init__(*args, **kwargs)
        self.v = v

    def F(self, x, n=None):
        beta = (self.v - 2) / 2
        return 1-gammainc(self.v/2, beta / (x ** 2))

    def Q(self, y, n=None):
        beta = (self.v - 2) / 2
        return (beta / gammaincinv(self.v / 2, 1-y)) ** 0.5


class Weibull(CDF):
    def __init__(self, l=1, k=1, *args, **kwargs):
        """ Weibull"""
        super(Weibull, self).__init__(*args, **kwargs)
        self.l = l
        self.k = k

    def F(self, x, n=None):
        return 1 - torch.exp(-(x/self.l)**self.k)

    def Q(self, y, n=None):
        return self.l*(-torch.log(1-y))**(1/self.k)


class Uniform(CDF):
    def __init__(self, l=0, r=1, *args, **kwargs):
        """ Uniform"""
        super(Uniform, self).__init__(*args, **kwargs)
        self.l = l
        self.r = r

    def F(self, x, n=None):
        return (x - self.l) / (self.r - self.l)

    def Q(self, y, n=None):
        return y * (self.r - self.l) + self.l


class Pareto(CDF):
    def __init__(self, s=0, a=1, *args, **kwargs):
        """ Uniform"""
        super(Pareto, self).__init__(*args, **kwargs)
        self.s = s
        self.a = a

    def F(self, x, n=None):
        return 1 - (self.s/x)**self.a

    def Q(self, y, n=None):
        return self.s/((1 - y)**(1/self.a))
