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







def loss_logp(params_unc, trace, obs, model, index_batch=None, device=None):
    transforms = get_transforms(trace)
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
            site["log_prob"] = site_log_p
            dims = tuple(range(1, site['log_prob'].dim()))
            if len(dims) > 0:
                logp += torch.sum(site['log_prob'], dim=dims).to(device)
                if name in transforms:
                    logp += -transforms[name].log_abs_det_jacobian(site['value'], params_unc[name]).sum(dims).to(device)
            else:
                logp += site['log_prob'].to(device)
                if name in transforms:
                    logp += -transforms[name].log_abs_det_jacobian(site['value'], params_unc[name]).to(device)
    logp[torch.isnan(logp)] = torch.tensor(-1e10, device=logp.device)
    return -logp




def gaussian_step(params, sigma):
    sigma_torch = {'cpu': torch.tensor(sigma, device='cpu', dtype=torch.float)}
    if torch.cuda.is_available():
        sigma_torch['cuda:0'] = torch.tensor(sigma, device='cuda', dtype=torch.float)
    proposal = {}
    for name, node in params.items():
        scale = torch.mul(sigma_torch[str(node.device)], torch.ones_like(node))
        proposal[name] = torch.distributions.Normal(loc=node, scale=scale).sample()
    return proposal


def sample_emcee_pond(nchains, niter, a, device):
    low = torch.zeros(nchains, device=device)
    high = torch.ones(nchains, device=device)
    r = torch.distributions.Uniform(low=low, high=high).sample(torch.Size([niter]))
    a_torch = torch.tensor(a, device=device, dtype=torch.float)
    return torch.div(torch.pow(torch.mul(a_torch - one.to(device), r) + one.to(device), two.to(device)), a_torch)


def emcee_step(params, pond_emcee, index_chains, arange, device):
    nchains = index_chains.shape[0]
    perm = torch.randint(0, nchains - 1, (nchains,), device=device, dtype=torch.long)
    index = torch.index_select(index_chains.view(-1), 0, perm + arange)
    proposal = {}
    _one = one.to(device)
    for name, node in params.items():
        shape = (-1, ) + (1, )*(len(node.shape) - 1)
        node_dev = node.to(device)
        proposal[name] = (torch.mul(pond_emcee.view(shape), node_dev) + torch.mul(_one - pond_emcee.view(shape), torch.index_select(node_dev, 0, index))).to(node.device)
    return proposal


def sgd_samples(model, nparams=50, niter=1000, data=None, times_train = None, times_cross=None, stochastic_gradient=True,
                burnin=None, p_batch=0.7, update_loss=10, device=None):
    trace = poutine.trace(model).get_trace(data)
    transforms = get_transform(trace)
    params_map = {}
    for name, node in trace.iter_stochastic_nodes():
        if name in transforms:
            params_map[name] = transforms[name](node['value'])
        else:
            params_map[name] = node['value']
        params_map[name].requires_grad_(True)

    ndim = len(dict_to_array(params_map, trace)) // nparams
    optimizer = torch.optim.Adam([{'params': [v for k, v in params_map.items() if k not in ['sigma', 'noise']],
                                   'lr': 1e-2},
                                  {'params': [v for k, v in params_map.items() if k in ['sigma', 'noise']],
                                   'lr': 1e-2}], lr=1e-2)
    optimizer.zero_grad()
    iter_range = torch.arange(niter, dtype=torch.long, device=device)
    loss_cross_niter = torch.empty(niter, nparams, dtype=torch.float, device=device)
    loss_batch_niter = torch.empty(niter, nparams, dtype=torch.float, device=device)

    size_batch = int(len(times_cross)*p_batch)
    times_batch = torch.arange(size_batch, device=device)
    times_random = torch.arange(size_batch, dtype=torch.float, device=device)

    zero_long = torch.tensor(0).to(device)
    progress = tqdm_notebook(iter_range)
    all_params = []
    for t in progress:

        # Compute and print loss
        if stochastic_gradient:
            _, select_batch = times_random.uniform_().sort()
        torch.index_select(times_train, 0, select_batch[:size_batch], out=times_batch)

        # loss
        loss_batch = loss_logp(params_map, trace, data, model, times_batch, device=device)
        loss_backward = torch.sum(loss_batch)

        with torch.no_grad():
            loss_cross = loss_logp(params_map, trace, data, model, times_cross, device=device)

        loss_cross_niter[t] = loss_cross
        loss_batch_niter[t] = loss_batch

        if zero_long.equal(t % update_loss):
            progress.set_description('{0:.5f}'.format(float(numpy(loss_cross.mean()))))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss_backward.backward(retain_graph=False)
        optimizer.step()

        #all_params += [{name: deepcopy(value.data.to('cpu')) for name, value in params_map.items()}]
        all_params += [{name: deepcopy(node['value'].data.to('cpu')) for name, node in trace.iter_stochastic_nodes()}]

        del loss_cross
        del loss_batch
    loss_cross_np = numpy(loss_cross_niter).copy()
    loss_batch_np = numpy(loss_batch_niter).copy()
    loss = {'_loss_cross': loss_cross_np.T, '_loss_batch': loss_batch_np.T}
    params_df = get_params_df_from_list_dict(all_params, loss)
    if burnin is None:
        burnin = niter//2
    params_df['_burnin'] = params_df['_niter'] > burnin
    params_df['_outlier'] = (~params_df.isna()).prod(axis=1) | (
                params_df['_loss_cross'] > np.percentile(params_df['_loss_cross'].unique(), 1))
    return params_df


def mcmc_samples(model, init, nparams=50, niter=1000, data=None, times_train = None, times_cross=None, emcee = True,
                 a_emcee = 1.5, sigma_gaussian=0.05, burnin=None, update_acc=10, device=None):

    trace = poutine.trace(model).get_trace(data)
    transforms = get_transform(trace)
    params_mcmc = get_params_dict_from_df(init, trace, device)
    for name, node in trace.iter_stochastic_nodes():
        if name in transforms:
            params_mcmc[name] = transforms[name](params_mcmc[name].to(node['value'].device))
        else:
            params_mcmc[name] = params_mcmc[name].to(node['value'].device)

    index_chains = np.empty((nparams, nparams - 1))
    for i in range(nparams):
        for j in range(nparams - 1):
            if i <= j:
                index_chains[i, j] = j + 1
            else:
                index_chains[i, j] = j
    index_chains = torch.tensor(index_chains, device=device, dtype=torch.long)
    nchains = index_chains.shape[0]
    emcee_arange = torch.arange(0, (nchains) * (nchains), nchains - 1, device=device)[:nchains]
    u = torch.distributions.Uniform(low=torch.zeros(nparams, device=device),
                                    high=torch.ones(nparams, device=device)).sample(torch.Size([niter])).log()
    pond_emcee = sample_emcee_pond(nparams, niter, a=a_emcee, device=device)

    ndim = len(dict_to_array(params_mcmc, trace)) // nparams
    iter_range = torch.arange(niter, dtype=torch.long, device=device)
    loss_cross_niter = torch.empty(niter, nparams, dtype=torch.float, device=device)
    loss_batch_niter = torch.empty(niter, nparams, dtype=torch.float, device=device)

    with torch.no_grad():
        loss_batch = loss_logp(params_mcmc, trace, data, model, times_train, device=device)
        loss_cross = loss_logp(params_mcmc, trace, data, model, times_cross, device=device)
        loss_batch[loss_batch != loss_batch] = np.infty

    zero_long = torch.tensor(0).to(device)
    zero = torch.tensor(0.0).to(device)
    one = torch.tensor(1.0).to(device)
    acc_rate = 0
    all_params = []

    progress = tqdm_notebook(iter_range)
    with torch.no_grad():
        for t in progress:

            if emcee:
                proposal = emcee_step(params_mcmc, pond_emcee[t], index_chains=index_chains, arange=emcee_arange, device=device)
            else:
                proposal = gaussian_step(params_mcmc, sigma=sigma_gaussian)

            loss_batch_prop = loss_logp(proposal, trace, data, model, times_train, device=device)
            loss_cross_prop = loss_logp(proposal, trace, data, model, times_cross, device=device)
            loss_batch_prop[loss_batch != loss_batch] = np.infty

            delta_loss = loss_batch - loss_batch_prop
            delta_loss[delta_loss != delta_loss] = 0.0
            if emcee:
                rho = torch.min(torch.add(torch.mul(ndim - one, pond_emcee[t].log()), delta_loss), zero)
            else:
                rho = torch.min(delta_loss, zero)

            index = (u[t] < rho)
            index_float = index.type(torch.float)

            # print(index)
            acc_rate += index_float.mean().to('cpu')

            # get new set of params
            for var, value in params_mcmc.items():
                params_mcmc[var][index] = proposal[var][index]
            all_params += [{name: node['value'].data.to(device) for name, node in trace.iter_stochastic_nodes()}]

            loss_batch[index] = loss_batch_prop[index]
            loss_cross[index] = loss_cross_prop[index]
            loss_cross_niter[t] = loss_cross
            loss_batch_niter[t] = loss_batch
            if zero_long.equal(t % update_acc):
                progress.set_description('{0:.5f}'.format(acc_rate / t.item()))

    loss_cross_np = numpy(loss_cross_niter).copy()
    loss_batch_np = numpy(loss_batch_niter).copy()
    loss = {'_loss_cross': loss_cross_np.T, '_loss_batch': loss_batch_np.T}
    params_df = get_params_df_from_list_dict(all_params, loss)
    if burnin is None:
        burnin = niter//2
    params_df['_burnin'] = params_df['_niter'] > burnin
    params_df['_outlier'] = (~params_df.isna()).prod(axis=1) | (
                params_df['_loss_cross'] > np.percentile(params_df['_loss_cross'].unique(), 1))
    return params_df

def sample_posterior(use_init = None):
    times_train = torch.arange(ndata, device=device)
    times_cross = torch.arange(ndata, ndata+ncross, device=device)
    if use_init is None:
        params_df = tpy.sgd_samples(model=sample_model, nparams=nparams, niter=sgd_niter, data=obs,
                                    times_train = times_train, times_cross=times_cross,
                                    stochastic_gradient = True, p_batch = 0.7, device=device_ll)
        tpy.plot_training(params_df[params_df._outlier], alpha=0.5, traces=True, confidence=True)
        plt.show()
        sgd_sample = params_df[params_df._burnin & params_df._outlier & (params_df._loss_cross < 1e10)].sort_values(by='_loss_cross').drop(['_nchain','_niter','_outlier','_burnin'], axis=1)[:nparams]
        _ = sgd_sample.hist(bins=nparams)
        plt.tight_layout()
        plt.show()
        init = sgd_sample
    else:
        init = use_init[:nparams]
    params_df = tpy.mcmc_samples(model=sample_model, init=init, nparams=nparams, niter=mcmc_niter,
                                 data=obs, times_train = times_train, times_cross=times_cross,
                                 emcee = True, a_emcee = 2.0, sigma_gaussian=0.02, device=device_ll)
    try:
        tpy.plot_training(params_df[params_df._outlier], alpha=0.5, traces=True, confidence=True)
        plt.show()
    except:
        pass

    posterior_df = params_df[params_df._burnin & params_df._outlier & (params_df._loss_cross < 1e10)].sort_values(by='_loss_batch').drop(['_nchain','_niter','_outlier','_burnin'], axis=1)
    _ = posterior_df.hist(bins=nparams)
    plt.tight_layout()
    plt.show()
    display(posterior_df.describe())
    return posterior_df