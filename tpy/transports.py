import torch
from .core import TpModule, resolve_inverse, shape, robust_cholesky
from .radials import NormGaussian
from .samplers import abc_samples
cholesky = robust_cholesky


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

    def logdetgradinv(self, t, y, sy=None):
        pass


class MarginalTransport(Transport):
    def __init__(self, marginal, *args, **kwargs):
        super(MarginalTransport, self).__init__(*args, **kwargs)
        self.marginal = marginal

    def forward(self, t, x, noise=None):
        #print(self.name, 'forward', noise)
        return self.marginal(t, x)

    def inverse(self, t, y, noise=None):
        #print(self.name, 'inverse', noise)
        return self.marginal.inverse(t, y)

    def posterior(self, t, x, obs_t=None, obs_y=None, generator=None, noise=False):
        ''' $T_{t}(x)$ '''
        return self.forward(t, x, noise=noise)

    def logdetgradinv(self, t, y, sy=None):
        return self.marginal.log_gradient_inverse(t, y).sum(dim=2).sum(dim=1)


class CovarianceTransport(Transport):
    def __init__(self, kernel, noise=None, pseudo_inputs=None, pseudo_outputs=None, *args, **kwargs):
        super(CovarianceTransport, self).__init__(*args, **kwargs)
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
        #print(self.name, self.kernel, 'forward', noise)
        '''$cho y$'''
        if noise:
            kernel = self.kernel_noise
        else:
            kernel = self.kernel
        return torch.matmul(cholesky(kernel(t)), x)

    def inverse(self, t, y, noise=True):
        #print(self.name, self.kernel, 'inverse', noise)
        '''$cho^{-1}y$'''
        if noise:
            kernel = self.kernel_noise
        else:
            kernel = self.kernel
        self.cholesky_cache = cholesky(kernel(t))
        return torch.triangular_solve(y, self.cholesky_cache, upper=False)[0]

    def posterior(self, t, x, obs_t, obs_y, generator=None, noise=False):
        #print(self.name, self.kernel, 'posterior', noise)
        nobs = obs_t.shape[0]
        if noise:
            kernel = self.kernel_noise
        else:
            kernel = self.kernel
        cov = self.kernel(obs_t, t)
        sigma = torch.cat([torch.cat([self.kernel_noise(obs_t), cov], dim=2),
                           torch.cat([cov.transpose(1, 2), kernel(t)], dim=2)], dim=1)
        cho = cholesky(sigma)
        cho_cross = cho[:, nobs:, :nobs]
        cho_obs = cho[:, :nobs, :nobs]
        cho_bar = cho[:, nobs:, nobs:]

        cho_solve = torch.triangular_solve(obs_y, cho_obs, upper=False)[0]
        return cho_cross.matmul(cho_solve) + torch.matmul(cho_bar, x)

    def logdetgradinv(self, t, y=None, sy=None, noise=True):
        if self.cholesky_cache is None:
            if noise:
                self.cholesky_cache = cholesky(self.kernel_noise(t))
            else:
                self.cholesky_cache = cholesky(self.kernel(t))
        return -torch.diagonal(self.cholesky_cache, dim1=1, dim2=2).log().sum(dim=1)


class Norm2Transport(Transport):
    ''' y = Q(F(|x|)) x / |x| '''
    def __init__(self, obj, ref=None, *args, **kwargs):
        super(Norm2Transport, self).__init__(*args, **kwargs)
        self.obj = obj
        if ref is None:
            ref = NormGaussian()
        self.ref = ref

    def norm_y(self, norm_x, n=None):
        return self.obj.Q(self.ref.F(norm_x, n), n)

    def norm_x(self, norm_y, n=None):
        return self.ref.Q(self.obj.F(norm_y, n), n)

    def scale(self, norm_y, n=None):
        return norm_y/self.norm_x(norm_y, n)

    def forward(self, t, x, noise=None):
        normx = x.norm(dim=-2)
        return self.norm_y(normx, n=t.shape[0])*x/normx

    def inverse(self, t, y, noise=None):
        normy = y.norm(dim=-2)[:, :, None]
        scale = self.norm_x(normy, n=t.shape[0])/normy
        return scale*y

    def posterior_norm_y(self, norm_x, n=None, n_obs=None, norm_y_obs=None):
        return self.obj.posterior_Q(self.ref.F(norm_x, n), n, n_obs, norm_y_obs)

    def posterior(self, t, x, obs_t, obs_y, generator=None, noise=False):
        normx = x.norm(dim=-2)
        normy_obs = obs_y.norm(dim=-2)
        post_normy = self.posterior_norm_y(normx, n=t.shape[0], n_obs=obs_t.shape[0], norm_y_obs=normy_obs)
        return post_normy*x/normx

    def logdetgradinv(self, t, y, sy=None, eps=1e-4):
        normy = y.norm(dim=-2)[:, :, None]
        normx = self.norm_x(normy, n=t.shape[0])
        normy = self.norm_y(normx, n=t.shape[0])

        scale = normy/normx
        dalpha = (self.norm_x(normy * (1 + eps)+eps, n=t.shape[0]) - self.norm_x(normy * (1 - eps)-eps, n=t.shape[0])) / (
                (normy+1) * 2*eps)
        #print(scale)
        #print(dalpha)
        scale[dalpha[:, 0, 0] != dalpha[:, 0, 0], 0, 0] = 1e-10
        dalpha[dalpha[:, 0, 0] != dalpha[:, 0, 0], 0, 0] = 1e-10
        #print(-(t.shape[0]-1)*scale[:, 0, 0].log() + dalpha[:, 0, 0].log())
        return -(t.shape[0]-1)*scale[:, 0, 0].log() + dalpha[:, 0, 0].log()


class RadialTransportOld(Transport):
    ''' y = Q(F(|x|)) x / |x| '''
    def __init__(self, mixture=None, obj=None, ref=None, *args, **kwargs):
        super(RadialTransportOld, self).__init__(*args, **kwargs)
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
            return resolve_inverse(lambda norm_x: self.mixture.Q(self.ref.F(norm_x, n)) * norm_x, norm_y, x0=torch.ones_like(norm_y)*n.sqrt())
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
            gen = lambda t0, n: self.forward(t0, generator.prior(t0, n))
            samples = abc_samples(gen, t, obs_t, obs_y[i], x.shape[1])
            posterior_norm_y[i, :, :] = samples
        return x[None, :, :]*(posterior_norm_y.norm(dim=1, keepdim=True)/x.norm(dim=0))

    def logdetgradinv(self, t, y):
        return 0