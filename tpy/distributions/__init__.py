import torch

import numpy as np
import pandas as pd
import scipy.linalg
from torch import nn
from pyro.distributions import Distribution, Bernoulli
from torch.distributions import biject_to, constraints
#from pyro.distributions.util import scale_tensor
from pyro import poutine






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


