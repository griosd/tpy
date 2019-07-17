import torch
import numpy as np
from tqdm import tqdm_notebook
from copy import deepcopy
from pyro import poutine
from .core import numpy
from torch.distributions import biject_to, constraints
from .utils import dict_to_array, get_params_df_from_list_dict, get_params_dict_from_df

one = torch.tensor(1.0)
two = torch.tensor(2.0)


def abc_samples(posterior, t, obs_t, obs_y, nsamples=100, pdarts=33, eps=5e-3, ntraits=1000):
    device = obs_y.device
    samples = torch.empty(t.shape[0], nsamples, device=device)
    obs_n = obs_t.shape[0]
    all_t = torch.cat([obs_t, t])
    obs_y_norm = obs_y.norm(dim=0)
    current_samples = 0
    darts = (nsamples - current_samples) * pdarts
    for i in range(ntraits):
        samples_darts = posterior(all_t, darts)[0]
        samples_darts_norm = samples_darts[:obs_n].norm(dim=0)
        samples_darts_index = (obs_y_norm * (1 - eps) < samples_darts_norm) & (samples_darts_norm < obs_y_norm * (1 + eps))
        samples_darts_n = samples_darts_index.sum().item()
        samples[:, current_samples: (current_samples+samples_darts_n)] = samples_darts[obs_n:, samples_darts_index][:, :min(nsamples-current_samples, samples_darts_n)]
        current_samples += samples_darts_n
        if current_samples >= nsamples:
            return samples
    print('incomplete samples:', current_samples, ' of ', nsamples)
    return samples


def get_transforms(trace):
    transforms = {}
    for name, node in trace.iter_stochastic_nodes():
        if node["fn"].support is not constraints.real:
            transforms[name] = biject_to(node["fn"].support)
    return transforms


def params_transform(params, transforms=None, trace=None):
    if transforms is None:
        transforms = get_transforms(trace)
    params_trans = {}
    for name in params.keys():
        if name in transforms:
            params_trans[name] = transforms[name].inv(params[name])
        else:
            params_trans[name] = params[name]
    return params_trans


def params_original(params_trans, transforms=None, trace=None):
    if transforms is None:
        transforms = get_transforms(trace)
    params = {}
    for name in params_trans.keys():
        if name in transforms:
            params[name] = transforms[name](params_trans[name])
        else:
            params[name] = params_trans[name]
    return params


def sgd_samples(tgp, params_model, niter=1000, start=None, optim=torch.optim.Rprop, lr=1e-2, psgd=0.5, pbatch=0.7, update_batch=100, update_tqdm=10):
    device = tgp.device
    obs_n = len(tgp.obs_t)
    size_batch = int(obs_n * pbatch)
    niter_sgd = int(niter * psgd)
    all_params = []
    all_loss = []

    params_model(start)
    trace = poutine.trace(params_model).get_trace()
    transforms = get_transforms(trace)
    params = params_model(start)
    params_map = params_transform(params, transforms)
    for name, param in params_map.items():
        param.requires_grad_()
    optimizer = optim(params_map.values(), lr=lr)
    optimizer.zero_grad()
    original_params = params_original(params_map, transforms)
    params_model(params=original_params)

    times_random = torch.empty(obs_n, device=device)
    _, select_batch = times_random.uniform_().sort()
    index_batch = select_batch[:size_batch]
    iter_range = torch.arange(niter, dtype=torch.long, device=device)
    progress = tqdm_notebook(iter_range)
    zero_long = torch.tensor(0).to(device)
    for t in progress:

        if t < niter_sgd:
            if zero_long.equal(t % update_batch):
                _, select_batch = times_random.uniform_().sort()
                index_batch = select_batch[:size_batch]
        else:
            index_batch = Ellipsis

        params = params_original(params_map, transforms)
        params_model(params=params)
        loss_nll = tgp.nll(index=index_batch)
        loss_backward = loss_nll.sum()

        all_params += [{k: v.detach().data.clone() for k, v in params.items()}]
        nll = tgp.nll(index=Ellipsis).detach().data.clone() + torch.cat([v for k,v in all_params[-1].items()], dim=1).sum(dim=1)[:, 0]*0
        # print(nll)
        all_loss += [nll]

        optimizer.zero_grad()
        loss_backward.backward(retain_graph=True)

        def closure():
            return loss_backward
        #print({k: (params[k], v.grad) for k, v in params_map.items() if v.grad is not None})
        optimizer.step(closure)

        if zero_long.equal(t % update_tqdm):
            last_loss = all_loss[-1]
            progress.set_description('{0:.5f}'.format(last_loss[last_loss==last_loss].min().item()))

    return all_params, all_loss


