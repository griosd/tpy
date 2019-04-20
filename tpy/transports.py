import torch
from .core import TpModule
from .radials import Chi, resolve_inverse


class Transport(TpModule):
    def __init__(self, *args, **kwargs):
        super(Transport, self).__init__(*args, **kwargs)

    def forward(self, t, x):
        pass

    def inverse(self, t, y):
        pass

    def logdetgradinv(self, t, y):
        pass


class ComposedTransport(Transport):
    def __init__(self, transports, *args, **kwargs):
        super(ComposedTransport, self).__init__(*args, **kwargs)
        self.transports = transports

    def forward(self, t, x):
        xi = x
        for T in self.transports:
            xi = T(t, xi)
        return xi

    def inverse(self, t, y):
        yi = y
        for T in reversed(self.transports):
            yi = T.inverse(t, yi)
        return yi


class ScatterTransport(Transport):
    def __init__(self, kernel, noise=None, pseudo_inputs=None, pseudo_outputs=None, *args, **kwargs):
        super(ScatterTransport, self).__init__(*args, **kwargs)
        self.kernel = kernel
        self.noise = noise
        self.pseudo_inputs = pseudo_inputs
        self.pseudo_outputs = pseudo_outputs

    def kernel_noise(self, t1, t2=None):
        if self.noise is None:
            return self.kernel(t1, t2)
        else:
            return self.kernel(t1, t2) + self.noise(t1, t2)

    def forward(self, t, x, noise=False):
        return torch.matmul(torch.cholesky(self.kernel(t)), x)

    def inverse(self, t, y, noise=True):
        return torch.potrs(y, torch.cholesky(self.kernel_noise(t)))


class WarpedTransport(Transport):
    def __init__(self, warped, *args, **kwargs):
        super(WarpedTransport, self).__init__(*args, **kwargs)
        self.warped = warped

    def forward(self, t, x):
        return self.warped(t, x)

    def inverse(self, t, y):
        return self.warped.inverse(t, y)


class RadialTransport(Transport):
    def __init__(self, obj, ref=None, *args, **kwargs):
        super(RadialTransport, self).__init__(*args, **kwargs)
        self.obj = obj
        if ref is None:
            ref = Chi()
        self.ref = ref

    def forward(self, t, x):
        return self.obj.Q(self.ref.F(x.norm(dim=0), x.shape[0])) * x

    def inverse(self, t, y):
        return resolve_inverse(self.forward, y)


