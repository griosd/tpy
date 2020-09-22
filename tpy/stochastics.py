import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from .core import TpModule, numpy
from .utils import plot2d, plot_text
from .transports import Transport

log2pi = math.log(2*math.pi)


class StochasticProcess(TpModule):

    def __init__(self, *args, **kwargs):
        super(StochasticProcess, self).__init__(*args, **kwargs)
        self.obs_t = None
        self.obs_y = None
        self.is_iid = False
        self.description = {}

    def obs(self, t, y):
        assert len(y.shape) == 3, "shape of obs_y="+str(y.shape)
        self.obs_t = t
        self.obs_y = y

    def nll(self, x):
        pass

    def prior(self, t, nsamples=1, noise=False):
        pass

    def posterior(self, t, nsamples=1, obs_t=None, obs_y=None, noise=False):
        pass

    def sample(self, t, n=1):
        if self.obs_t is None:
            return self.prior(t, n)
        else:
            return self.posterior(t, n)

    def plot(self, t, samples=None, subsamples=Ellipsis, mean=True, quantiles=[0.025, 0.975], obs=True, hidden=None,
             logp=None, alpha=0.2, xlim=None, ylim=None, loc='best', ncol=3, palette='Greens', *args, **kwargs):
        cmap = plt.get_cmap(palette)
        color = cmap(1.0)
        if samples is None:
            samples = t
            t = torch.arange(samples.shape[0])
        if len(samples.shape) > 2:
            if samples.shape[0] == 1:
                samples = samples[0]
            else:
                samples = samples.view(samples.shape[1], -1)
        if subsamples is not None:
            if type(subsamples) is int:
                subsamples = torch.randperm(samples.shape[1])[:subsamples]
            plot2d(t, samples[:, subsamples], color=cmap(0.9), alpha=alpha, lw=1,  *args, **kwargs)
        if mean:
            smean = samples.mean(dim=-1)
            plot2d(t, smean, color=color, lw=3, label='Mean')
            # plot2d(t, smean, color='w', lw=1)
        if quantiles:
            np_samples = numpy(samples)
            plot2d(t, np.percentile(np_samples, 100*quantiles[0], axis=1), color=cmap(0.8), alpha=0.8, lw=1.5, ls='--',
                   label=(str(int(100*(max(quantiles)-min(quantiles)))))+'% CI')
            for q in quantiles[1:]:
                plot2d(t, np.percentile(np_samples, 100*q, axis=1), color=cmap(0.8), alpha=0.8, lw=1.5, ls='--')
            quantile_down = np.percentile(np_samples, 100*min(quantiles), axis=1)
            quantile_up = np.percentile(np_samples, 100*max(quantiles), axis=1)
            plt.fill_between(numpy(t), quantile_up, quantile_down, color=cmap(0.8), alpha=0.1)

        if xlim is None:
            plt.xlim([t.min().item(), t.max().item()])
        else:
            plt.xlim(xlim)
        if ylim is None:
            plt.ylim([samples.min().item(), samples.max().item()])
        else:
            plt.ylim(ylim)
        if hidden is not None:
            plot2d(t, hidden, color='k', label='Hidden Process')
        if obs and not (self.obs_t is None):
            if len(self.obs_y.shape) > 2:
                for y in self.obs_y:
                    plot2d(self.obs_t, y, '.', color='k', ms=10, label='Observations')
            else:
                plot2d(self.obs_t, self.obs_y, '.', color='k', ms=10, label='Observations')

        if loc is not None and len(self.description) > 0:
            title = self.description['title']
            if logp is not None:
                title += ' (logp: {:.3f})'.format(logp)
            plot_text(title, self.description['x'], self.description['y'], loc=loc, ncol=ncol)

    def describe(self, title=None, x=None, y=None, text=None):
        """
        Set descriptions to the model
        Args:
            title (str): The title of the process
            x (str):the label for the dependient variable
            y (str):the label for the independient variable
            text (str): aditional text if necessary.
        """

        if title is not None:
            self.description['title'] = title
        if title is not None:
            self.description['x'] = x
        if title is not None:
            self.description['y'] = y
        if title is not None:
            self.description['text'] = text


class GWNP(StochasticProcess):
    def __init__(self, *args, **kwargs):
        super(GWNP, self).__init__(*args, **kwargs)
        self.dtype = torch.float32
        self.is_iid = True

    def prior(self, t, nsamples=1, noise=False):
        return torch.empty((t.shape[0], nsamples), dtype=self.dtype, device=self.device).normal_()

    def posterior(self, t, nsamples=1, obs_t=None, obs_y=None, noise=False):
        return torch.empty((t.shape[0], nsamples), dtype=self.dtype, device=self.device).normal_()

    def nll(self, x):
        """x.shape = (nparams, nobs, noutput)"""
        return 0.5*(x.shape[1]*log2pi + (x**2).sum(dim=1)).sum(dim=1)


class TP(StochasticProcess):
    def __init__(self, generator, transport, *args, **kwargs):
        super(TP, self).__init__(*args, **kwargs)
        self.generator = generator
        if isinstance(transport, Transport):
            self.transport = [transport]
        else:
            self.transport = transport

    def obs(self, t, y):
        self.obs_t = t
        self.obs_y = y

    def forward(self, t, x, noise=False):
        xi = x
        for T in self.transport:
            xi = T.forward(t, xi, noise=noise)
        return xi

    def inverse(self, t, y, noise=True, return_inv=True, return_list=False):
        list_inv_y = [y]
        for T in reversed(self.transport[1:]):
            list_inv_y.append(T.inverse(t, list_inv_y[-1], noise=noise))
        if return_inv:
            list_inv_y.append(self.transport[0].inverse(t, list_inv_y[-1], noise=noise))
            yi = list_inv_y[-1]
            if return_list:
                list_inv_y.reverse()
                return yi, list_inv_y
            else:
                return yi
        elif return_list:
            list_inv_y.reverse()
            return list_inv_y

    def logdetgradinv(self, t, list_obs_y):
        r = self.transport[0].logdetgradinv(t, y=list_obs_y[1], sy=list_obs_y[0])
        # print(self.transport[0], self.transport[0].logdetgradinv(t, y=list_obs_y[1], sy=list_obs_y[0]))
        for i in range(1, len(self.transport)):
            r += self.transport[i].logdetgradinv(t, y=list_obs_y[i+1], sy=list_obs_y[i])
            # print(self.transport[i], self.transport[i].logdetgradinv(t, y=list_obs_y[i+1], sy=list_obs_y[i]))
        return r

    @property
    def obs_x(self):
        return self.inverse(self.obs_t, self.obs_y, noise=True)

    def prior(self, t, nsamples=1, noise=False):
        return self.forward(t, self.generator.prior(t, nsamples=nsamples), noise=noise)

    def posterior(self, t,  nsamples=1, obs_t=None, obs_y=None, noise=False):
        if obs_t is None:
            obs_t = self.obs_t
        if obs_y is None:
            obs_y = self.obs_y
        list_obs_y = self.inverse(obs_t, obs_y, return_inv=False, return_list=True, noise=True)
        xi = self.generator.posterior(t, nsamples=nsamples)
        for i, T in enumerate(self.transport):
            xi = T.posterior(t, xi, obs_t, list_obs_y[i], generator=self.generator, noise=noise)
        return xi

    def nll(self, t=None, y=None, index=Ellipsis):
        if t is None:
            t = self.obs_t[index]
        if y is None:
            y = self.obs_y[:, index, :]
        inverse_y, list_obs_y = self.inverse(t, y, return_inv=True, return_list=True)
        return self.generator.nll(inverse_y) - self.logdetgradinv(t, list_obs_y=list_obs_y)


class TGP(TP):
    def __init__(self, transport, *args, **kwargs):
        super(TGP, self).__init__(generator=GWNP(), transport=transport, *args, **kwargs)
