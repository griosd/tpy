import torch
import gc as _gc
from torch import nn

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