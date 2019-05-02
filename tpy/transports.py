import torch
from .core import TpModule, resolve_inverse, shape
from .radials import NormGaussian
from .samplers import abc_samples


class Transport(TpModule):
    def __init__(self, *args, **kwargs):
        super(Transport, self).__init__(*args, **kwargs)

    def forward(self, t, x, noise=False):
        ''' $T_{t}(x)$ '''
        pass

    def inverse(self, t, y, noise=True):
        ''' $T_{t}^{-1}(y)$ '''
        pass

    def prior(self, t, nsamples=1, generator=None, noise=False):
        ''' $T_{t}(x)$ '''
        return self.forward(t, generator.prior(t, nsamples=nsamples), noise=noise)

    def posterior(self, t, x, obs_t, obs_y, generator, noise=False):
        ''' $T_{t}(x)$ '''
        pass

    def logdetgradinv(self, t, y):
        pass


class WarpedTransport(Transport):
    def __init__(self, warped, *args, **kwargs):
        super(WarpedTransport, self).__init__(*args, **kwargs)
        self.warped = warped

    def forward(self, t, x, noise=None):
        return self.warped(t, x)

    def inverse(self, t, y, noise=None):
        return self.warped.inverse(t, y)

    def posterior(self, t, x, obs_t=None, obs_y=None, generator=None, noise=False):
        ''' $T_{t}(x)$ '''
        return self.forward(t, x)

    def logdetgradinv(self, t, y):
        return -self.warped.gradient_inverse(t, y).log().sum(dim=2).sum(dim=1)


class ScatterTransport(Transport):
    def __init__(self, kernel, noise=None, pseudo_inputs=None, pseudo_outputs=None, *args, **kwargs):
        super(ScatterTransport, self).__init__(*args, **kwargs)
        self.kernel = kernel
        self.noise = noise
        self.pseudo_inputs = pseudo_inputs
        self.pseudo_outputs = pseudo_outputs
        self.cholesky_cache = None

    def kernel_noise(self, t1, t2=None):
        if self.noise is None:
            return self.kernel(t1, t2)
        else:
            return self.kernel(t1, t2) + self.noise(t1, t2)

    def forward(self, t, x, noise=False):
        '''$cho y$'''
        if noise:
            kernel = self.kernel_noise
        else:
            kernel = self.kernel
        return torch.matmul(torch.cholesky(kernel(t)), x)

    def inverse(self, t, y, noise=True):
        '''$cho^{-1}y$'''
        if noise:
            kernel = self.kernel_noise
        else:
            kernel = self.kernel
        self.cholesky_cache = torch.cholesky(kernel(t))
        return torch.triangular_solve(y, self.cholesky_cache, upper=False)[0]

    def posterior(self, t, x, obs_t, obs_y, generator=None, noise=False):
        nobs = obs_t.shape[0]
        cat_t = torch.cat([obs_t, t])
        if noise:
            kernel = self.kernel_noise
        else:
            kernel = self.kernel
        cho = torch.cholesky(kernel(cat_t))
        cho_cross = cho[:, nobs:, :nobs]
        cho_obs = cho[:, :nobs, :nobs]
        cho_bar = cho[:, nobs:, nobs:]
        return cho_cross.matmul(torch.triangular_solve(obs_y, cho_obs, upper=False)[0]) + torch.matmul(cho_bar, x)

    def logdetgradinv(self, t, y=None, noise=True):
        if self.cholesky_cache is None:
            if noise:
                self.cholesky_cache = torch.cholesky(self.kernel_noise(t))
            else:
                self.cholesky_cache = torch.cholesky(self.kernel(t))
        return torch.diagonal(self.cholesky_cache, dim1=1, dim2=2).log().sum(dim=1)


class RadialTransport(Transport):
    ''' y = Q(F(|x|)) x / |x| '''
    def __init__(self, mixture=None, obj=None, ref=None, *args, **kwargs):
        super(RadialTransport, self).__init__(*args, **kwargs)
        self.obj = obj
        self.mixture = mixture
        if ref is None:
            ref = NormGaussian()
        self.ref = ref

    def norm_y(self, norm_x, n=None):
        if self.obj is None:
            return self.scale(norm_x, n) * norm_x
        else:
            return self.obj.Q(self.ref.F(norm_x, n), n)

    def norm_x(self, norm_y, n=None):
        if self.obj is None:
            return resolve_inverse(lambda norm_x: self.mixture.Q(self.ref.F(norm_x, n)) * norm_x, norm_y)
        else:
            return self.ref.Q(self.obj.F(norm_y, n), n)

    def scale(self, norm_x, n=None):
        if self.obj is None:
            return self.mixture.Q(self.ref.F(norm_x, n))
        else:
            return self.norm_y(norm_x, n)/norm_x

    def forward(self, t, x, noise=None):
        return self.scale(x.norm(dim=0), shape(x)) * x

    def inverse(self, t, y, noise=None):
        norm_y = y.norm(dim=0)
        return self.norm_x(norm_y, shape(y)) * y / norm_y

    def posterior(self, t, x, obs_t, obs_y, generator, noise=False):
        if len(obs_y.shape) > 2:
            nparams = obs_y.shape[0]
        else:
            nparams = 1
            obs_y = obs_y[None, :, :]
        posterior_norm_y = torch.empty((nparams, x.shape[0], x.shape[1]), device=x.device)
        for i in range(nparams):
            posterior_norm_y[i, :, :] = abc_samples(lambda t0, n: self.forward(t0, generator.prior(t0, n))[i],
                                                    t, obs_t, obs_y[i], x.shape[1])
        return posterior_norm_y.norm(dim=1)*x/x.norm(dim=0)

    def logdetgradinv(self, t, y):
        pass

