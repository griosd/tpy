import math
import torch
import matplotlib.pyplot as plt
from .core import TpModule
from .utils import plot2d
from .transports import Transport, ComposedTransport

log2pi = math.log(2*math.pi)


class StochasticProcess(TpModule):
    def nll(self, x):
        pass

    def sample(self, t, n=1):
        pass

    def plot(self, t, samples=None, mean=True, color='g', alpha=0.2, xlim=None, ylim=None, *args, **kwargs):
        if samples is None:
            samples = t
            t = torch.arange(samples.shape[0])
        plot2d(t, samples, color=color, alpha=alpha, *args, **kwargs)
        if mean:
            smean = samples.mean(dim=-1)
            plot2d(t, smean, color=color, lw=2)
            plot2d(t, smean, color='w', lw=1)

        if xlim is None:
            plt.xlim([t.min().item(), t.max().item()])
        else:
            plt.xlim(xlim)
        if ylim is None:
            plt.ylim([samples.min().item(), samples.max().item()])
        else:
            plt.ylim(ylim)


class GP(StochasticProcess):
    def __init__(self, *args, **kwargs):
        super(GP, self).__init__(*args, **kwargs)
        self.dtype = torch.float32

    def sample(self, t, nsamples=1):
        return torch.empty((t.shape[0], nsamples), dtype=self.dtype, device=self.device).normal_()

    def nll(self, x):
        return -0.5*(x.shape[-1]*log2pi + torch.bmm(x,x))


class TP(StochasticProcess):
    def __init__(self, generator, transport, *args, **kwargs):
        super(TP, self).__init__(*args, **kwargs)
        self.generator = generator
        if isinstance(transport, Transport):
            self.transport = transport
        else:
            self.transport = ComposedTransport(transport)

    def sample(self, t, nsamples=1):
        return self.transport(t, self.generator.sample(t, nsamples))

    def nll(self, x):
        return self.eta.nll(self.transport.inverse(x)) + self.transport.logdetgradinv(x)


class TGP(TP):
    def __init__(self, transport, *args, **kwargs):
        super(TGP, self).__init__(generator=GP(), transport=transport, *args, **kwargs)