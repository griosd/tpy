import torch
import gc as _gc
import pyro
from torch import nn
from torch.nn import MSELoss

use_cuda = True


def cuda(use=True):
    global use_cuda
    use_cuda = use


def cpu(use=True):
    global use_cuda
    use_cuda = not use


def device():
    return torch.device("cuda") if torch.cuda.is_available() and use_cuda else torch.device("cpu")


def nan_to_num(t, num=0):
    t[torch.isnan(t)] = num
    return t


def gc():
    _gc.collect()
    torch.cuda.empty_cache()


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


def numpy(tensor):
    try:
        return tensor.numpy()
    except:
        try:
            return tensor.detach().numpy()
        except:
            try:
                return tensor.detach().cpu().numpy()
            except:
                return tensor


def resolve_inverse(f, y, x0=None, loss=MSELoss(), prop=True, tol=1e-5, max_iters=10000, *args, **kwargs):
    '''return x so f(x) = y'''
    if x0 is None:
        x = y.clone()
    else:
        x = x0.clone()
    x.requires_grad_(True)
    if prop:
        one = torch.tensor(1.0).to(x.device)
        cost = lambda f, y: loss(f / y, one)
    else:
        cost = loss
    optimizer = torch.optim.Rprop([x]) #RMSprop Adadelta
    for t in range(max_iters):
        optimizer.zero_grad()
        F = cost(f(x, *args, **kwargs), y)
        F.backward()
        optimizer.step()
        if F < tol:
            print(t, F)
            break
    return x


def shape(x, dim=0):
    return torch.as_tensor(x.shape[dim], dtype=torch.float32, device=x.device)


class NonInformative(pyro.distributions.Uniform):
    def log_prob(self, value):
        return torch.zeros_like(value)


def parameter(name, shape=(1,1,1), device=None):
    if device is None:
        device = device_fn()
    p = NonInformative(torch.zeros(shape, device=device), torch.ones(shape, device=device))
    p.name = name
    return p