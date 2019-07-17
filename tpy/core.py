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


def resolve_inverse(f, y, x0=None, loss=MSELoss(), prop=False, tol=1e-4, max_iters=10000, *args, **kwargs):
    '''return x so f(x) = y'''
    if x0 is None:
        x = y.clone()
    else:
        x = x0.clone()
    tol_F = tol*y.mean()
    x.requires_grad_(True)
    if prop:
        one = torch.tensor(1.0).to(x.device)
        cost = lambda p, y: loss(p / y, one)
    else:
        cost = loss
    optimizer = torch.optim.Rprop([x]) #RMSprop Adadelta
    for t in range(max_iters):
        optimizer.zero_grad()
        F = cost(f(x, *args, **kwargs), y)
        F.backward()
        optimizer.step()
        if F < tol_F:
            break
    if F > tol_F:
        print('loss/tol:', F, '/', tol_F)
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


def robust_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    mask = torch.eye(A.shape[1], A.shape[2], device=A.device).byte()
    for i in range(A.shape[0]):
        if A[i, 0, 0] == A[i, 0, 0]:
            pass
        else:
            A[i].zero_().masked_fill_(mask, 1e-10)

    Amax = A.diagonal(dim1=1, dim2=2).mean(dim=1)[:,None,None]
    A = A/Amax
    try:
        L = torch.cholesky(A, upper=upper, out=out)
        # TODO: Remove once fixed in pytorch (#16780)
        if A.dim() > 2 and A.is_cuda:
            if torch.isnan(L if out is None else out).any():
                raise RuntimeError

        return L*(Amax**0.5)
    except RuntimeError as e:
        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        jitter_ori = jitter
        Aprime = A.clone()
        for i in range(5):
            jitter = jitter_ori * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter)
            try:
                L = torch.cholesky(Aprime, upper=upper, out=out)
                # TODO: Remove once fixed in pytorch (#16780)
                if A.dim() > 2 and A.is_cuda:
                    if torch.isnan(L if out is None else out).any():
                        raise RuntimeError("singular")
                #warnings.warn(f"A not p.d., added jitter of {jitter} to the diagonal", RuntimeWarning)
                return L*(Amax**0.5)
            except RuntimeError:
                continue

        raise e
