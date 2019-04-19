import torch
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np
import pandas as pd
import scipy.linalg
from torch import nn
from pyro.distributions import Distribution, Bernoulli
from torch.distributions import biject_to, constraints
#from pyro.distributions.util import scale_tensor
from pyro import poutine

use_cuda = True


def cuda(use=True):
    global use_cuda
    use_cuda = use


def cpu(use=True):
    global use_cuda
    use_cuda = not use


def device():
    return torch.device("cuda") if torch.cuda.is_available() and use_cuda else torch.device("cpu")


def numpy(tensor):
    try:
        return tensor.numpy()
    except:
        try:
            return tensor.detach().numpy()
        except:
            return tensor.detach().cpu().numpy()


def nan_to_num(t, num=0):
    t[torch.isnan(t)] = num
    return t


device_fn = device


class TpModule(nn.Module):
    def __init__(self, device=None, name=None):
        super(TpModule, self).__init__()
        if device is None:
            device = device_fn()
        self._device = device.type
        if name is None:
            self.name = str(type(self)).split('.')[-1]
        else:
            self.name = name

    def forward(self, *input):
        pass

    def to(self, device):
        self._device = device.type
        return super(TpModule, self).to(device)

    @property
    def device(self):
        return torch.device(self._device)


class MixtureDistribution(Distribution):
    def __init__(self, mix1, mix2, p=None):
        self.mix1 = mix1
        self.mix2 = mix2
        self.p = Bernoulli(p)

    def log_prob(self, x):
        lp1 = self.mix1.log_prob(x)
        lp2 = self.mix2.log_prob(x)
        p1 = self.p.mean*torch.exp(lp1)
        p2 = (1-self.p.mean)*torch.exp(lp2)
        p = torch.log(p1 + p2)
        pj = torch.log(self.p.mean)+lp1 + torch.log(1-self.p.mean)+lp2
        mask = torch.isfinite(p)
        p[~mask] = pj[~mask]
        return p

    def sample(self, n_samples=None):
        if n_samples is None:
            p = self.p.sample()
            return p*self.mix1.sample()+(1-p)*self.mix2.sample()
        else:
            p = self.p.sample(n_samples)
            return p*self.mix1.sample(n_samples)+(1-p)*self.mix2.sample(n_samples)


class TransportFamily(Distribution):
    pass


class ConcatDistribution(TransportFamily):
    def __init__(self, generators, shapes=None):
        self.generators = generators
        self.shapes = shapes
        if self.shapes is None:
            self.shapes = [1 for _ in self.generators]

    @property
    def shape(self):
        return torch.Size([torch.sum(torch.tensor(self.shapes))])

    @property
    def zip(self):
        return zip(self.generators, self.shapes)

    def sample(self, n_samples=None):
        if n_samples is None:
            return torch.cat([d.sample((s,)) for d, s in self.zip])
        else:
            return torch.cat([d.sample((s, n_samples)) for d, s in self.zip]).t()

    def log_prob(self, x):
        lp = 0
        for xi, d, s in zip(torch.split(x, self.shapes, dim=-1), self.generators, self.shapes):
            lp += d.log_prob(xi).sum(dim=-1)
        return lp


class MatrixSquareRoot(Function):
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


sqrtm = MatrixSquareRoot.apply


def bsqrtm(batch):
    if len(batch.shape) == 2:
        return sqrtm(batch)
    else:
        return torch.stack([sqrtm(m) for m in batch])


def bgesv(B, A):
    return torch.stack([torch.gesv(b, a)[0] for b,a in zip(B,A)])


def blogdet(A):
    return torch.stack([torch.logdet(a) for a in A])


def get_transform(trace):
    transforms = {}
    for name, node in trace.iter_stochastic_nodes():
        if node["fn"].support is not constraints.real:
            transforms[name] = biject_to(node["fn"].support).inv
    return transforms


def loss_logp(params_unc, trace, obs, model, index_batch=None, device=None):
    transforms = get_transform(trace)
    params_const = {}
    for name, node in trace.iter_stochastic_nodes():
        if name in transforms:
            params_const[name] = transforms[name].inv(params_unc[name])
        else:
            params_const[name] = params_unc[name]+0
        node['value'] = params_const[name]

    trace_poutine = poutine.trace(poutine.replay(model, trace=trace))
    if index_batch is None:
        trace_poutine(obs)
    else:
        trace_poutine(obs[index_batch])
    trace = trace_poutine.trace
    logp = 0
    for name, site in trace.nodes.items():
        if site["type"] == "sample":
            args, kwargs = site["args"], site["kwargs"]
            site_log_p = site["fn"].log_prob(site["value"], *args, **kwargs)
            #site_log_p = scale_tensor(site_log_p, site["scale"])
            site["log_prob"] = site_log_p
            dims = tuple(range(1, site['log_prob'].dim()))
            if len(dims)>0:
                logp += torch.sum(site['log_prob'], dim=dims).to(device)
                if name in transforms:
                    logp += -transforms[name].log_abs_det_jacobian(site['value'], params_unc[name]).sum(dims).to(device)
            else:
                logp += site['log_prob'].to(device)
                if name in transforms:
                    logp += -transforms[name].log_abs_det_jacobian(site['value'], params_unc[name]).to(device)
    logp[torch.isnan(logp)] = torch.tensor(-1e10, device=logp.device)
    return -logp


