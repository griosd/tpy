import torch
import scipy
import numpy as np
from .core import TpModule, numpy
from torch.autograd import Variable
from math import pi

pi2 = 2*pi


class Kernel(TpModule):
    def __init__(self, *args, **kwargs):
        super(Kernel, self).__init__(*args, **kwargs)

    def k(self, x1, x2):
        pass

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        return self.k(x1.view(-1, 1), x2.view(1, -1))

    def __mul__(self, other):
        if issubclass(type(other), Kernel):
            return KernelProd(self, other)
        else:
            return KernelScale(self, other)
    __imul__ = __mul__

    def __rmul__(self, other):
        if issubclass(type(other), Kernel):
            return KernelProd(other, self)
        else:
            return KernelScale(self, other)

    def __add__(self, other):
        if issubclass(type(other), Kernel):
            return KernelSum(self, other)
        else:
            return KernelShift(self, other)
    __iadd__ = __add__

    def __radd__(self, other):
        if issubclass(type(other), Kernel):
            return KernelSum(other, self)
        else:
            return KernelShift(self, other)


class KernelOperation(Kernel):
    def __init__(self, _k: Kernel, _element, *args, **kwargs):
        super(Kernel, self).__init__(*args, **kwargs)
        self.kernel = _k
        self.element = _element


class KernelScale(KernelOperation):
    def k(self, x1, x2):
        return self.element * self.kernel.k(x1, x2)


class KernelShift(KernelOperation):
    def k(self, x1, x2):
        return self.element + self.kernel.k(x1, x2)


class KernelComposition(Kernel):
    def __init__(self,  _k1: Kernel, _k2: Kernel, *args, **kwargs):
        super(Kernel, self).__init__(*args, **kwargs)
        self.kernel1 = _k1
        self.kernel2 = _k2


class KernelProd(KernelComposition):
    def k(self, x1, x2):
        return self.kernel1.k(x1, x2) * self.kernel2.k(x1, x2)


class KernelSum(KernelComposition):
    def k(self, x1, x2):
        return self.kernel1.k(x1, x2) + self.kernel2.k(x1, x2)


class WN(Kernel):
    def __init__(self, var=None, *args, **kwargs):
        super(WN, self).__init__(*args, **kwargs)
        self.var = var

    def k(self, x1, x2):
        return self.var * x1.eq(x2).float()


class BW(Kernel):
    def __init__(self, var=None, *args, **kwargs):
        super(BW, self).__init__(*args, **kwargs)
        self.var = var

    def k(self, x1, x2):
        return torch.mul(self.var, torch.min(x1, x2))


class SE(Kernel):
    def __init__(self, var=None, scale=None, *args, **kwargs):
        super(SE, self).__init__(*args, **kwargs)
        self.var = var
        self.scale = scale

    def k(self, x1, x2):
        return self.var * torch.exp(-((x1 - x2) ** 2) / (self.scale ** 2))


class RQ(Kernel):
    def __init__(self, var=None, scale=None, alpha=None, *args, **kwargs):
        super(RQ, self).__init__(*args, **kwargs)
        self.var = var
        self.scale = scale
        self.alpha = alpha

    def k(self, x1, x2):
        return self.var * torch.pow(1 + ((x1 - x2) ** 2) / ((self.scale ** 2)*self.alpha), -self.alpha)


class SIN(Kernel):
    def __init__(self, var=None, scale=None, period=None, *args, **kwargs):
        super(SIN, self).__init__(*args, **kwargs)
        self.var = var
        self.scale = scale
        self.period = period

    def k(self, x1, x2):
        return self.var * torch.exp(-(torch.sin((x1 - x2) / self.period) ** 2) / (self.scale ** 2))


class COS(Kernel):
    def __init__(self, var=None, period=None, *args, **kwargs):
        super(COS, self).__init__(*args, **kwargs)
        self.var = var
        self.period = period

    def k(self, x1, x2):
        return self.var * torch.cos(pi2*(x1 - x2) / self.period)


class POL(Kernel):
    def __init__(self, var=None, bias=0, p=1, *args, **kwargs):
        super(POL, self).__init__(*args, **kwargs)
        self.bias = bias
        self.var = var
        self.p = p

    def k(self, x1, x2):
        return torch.mul(self.var, (self.bias+torch.mul(x1, x2))**self.p)


class SM(Kernel):
    def __init__(self, var=None, scale=None, freq=None, *args, **kwargs):
        super(SM, self).__init__(*args, **kwargs)
        self.var = var
        self.scale = scale
        self.freq = freq

    def k(self, x1, x2):
        return self.var * torch.exp(-((x1 - x2) ** 2) / (self.scale ** 2)) * torch.cos((x1 - x2) * self.freq)


class MatrixSquareRoot(torch.autograd.Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = numpy(input)
        try:
            sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).type_as(input)
        except Exception as e:
            print(m, e)
            sqrtm = input*1e6
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_variables
            sqrtm = numpy(sqrtm.data).astype(np.float_)
            gm = numpy(grad_output.data).astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).type_as(grad_output.data)
        return Variable(grad_input)


_sqrtm = MatrixSquareRoot.apply


def sqrtm(inputs):
    '''Square root of a positive definite matrix'''
    if len(inputs.shape) == 2:
        return _sqrtm(inputs)
    else:
        return torch.stack([_sqrtm(m) for m in inputs])
