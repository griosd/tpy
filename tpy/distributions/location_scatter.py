import torch
from .. import TransportFamily, TpModule, sqrtm, bgesv, blogdet, DictObj


class LocationScatter(TransportFamily):
    def __init__(self, generator, location=None, scatter=None, covariance=None):
        self.generator = generator
        self.location = location
        self.scatter = scatter
        if covariance is not None:
            self.scatter = sqrtm(covariance)

    @property
    def covariance(self):
        return torch.matmul(self.scatter.transpose(-1, -2), self.scatter)

    def update_scatter(self):
        self.scatter = sqrtm(self.covariance)

    def sample(self, n_samples=None):
        if n_samples is None:
            n_samples = 1
        r = torch.matmul(self.generator.sample(n_samples), self.scatter)
        if len(r.shape) == 3:
            r += self.location.unsqueeze(1)
        else:
            r += self.location
        if r.shape[0] == 1:
            r = r.squeeze(0)
        return r

    def log_prob(self, x):
        if len(self.scatter.shape) == 2:
            solve, _ = torch.gesv((x-self.location).t(), self.scatter)
            return self.generator.log_prob(solve.t()).sum() - len(x)*torch.logdet(self.scatter)
        try:
            solve = bgesv((x-self.location.unsqueeze(1)).transpose(1, 2), self.scatter).transpose(1, 2)
            return self.generator.log_prob(solve).sum(dim=-1) - len(x)*blogdet(self.scatter)
        except Exception as e:
            print(e)
            return self.generator.log_prob(x).sum(dim=-1)-1e10

    def cost_w2(self, batch, weights=None):
        if type(batch) in [LocationScatter, LocationCovariance]:
            batch = [DictObj(location=loc, scatter=s, covariance=cov) for loc, s, cov in zip(batch.location, batch.scatter, batch.covariance)]
        if weights is None:
            weights = torch.ones(len(batch))/torch.tensor(len(batch), dtype=torch.float)
            weights = weights.to(self.scatter.device)
        cost = 0
        for other, w in zip(batch, weights):
            c = sqrtm(torch.matmul(self.scatter, torch.matmul(other.covariance, self.scatter)))
            cost += w*(torch.sum((self.location-other.location)**2) + torch.sum(self.scatter**2 + other.scatter**2) - 2*torch.trace(c))
        return cost

    def sgd_step(self, batch, weights=None, step_size=1, update_scatter=True):
        if type(batch) in [LocationScatter, LocationCovariance]:
            batch = [DictObj(location=loc, scatter=s, covariance=cov) for loc, s, cov in zip(batch.location, batch.scatter, batch.covariance)]
        if weights is None:
            weights = torch.ones(len(batch))/torch.tensor(len(batch), dtype=torch.float)
            weights = weights.to(self.scatter.device)
        step_loc = 0
        step_cov = 0
        for other, w in zip(batch, weights):
            step_loc += w*other.location
            step_cov += w*sqrtm(torch.matmul(self.scatter, torch.matmul(other.covariance, self.scatter)))
        step_cov = (1 - step_size) * self.covariance + step_size * step_cov
        self.location = (1 - step_size)*self.location + step_size * step_loc
        inv_scatter = torch.inverse(self.scatter)
        self.scatter = sqrtm(torch.matmul(inv_scatter, torch.matmul(torch.matmul(step_cov,step_cov), inv_scatter)))
        if update_scatter:
            self.update_scatter()


class LocationCovariance(TransportFamily):
    def __init__(self, generator, location=None, scatter=None, covariance=None):
        self.generator = generator
        self.location = location
        self.covariance = covariance
        if scatter is not None:
            self.covariance = torch.matmul(scatter, scatter)

    @property
    def scatter(self):
        return sqrtm(self.covariance)

    def sample(self, n_samples=None):
        if n_samples is None:
            s = self.generator.sample()
        else:
            s = self.generator.sample(n_samples).t()
        return self.location + torch.matmul(self.scatter, s)

    def log_prob(self, x):
        solve, _ = bgesv(x-self.location, self.scatter)
        return self.generator.log_prob(solve) + torch.logdet(self.scatter)

    def cost_w2(self, batch, weights=None):
        if type(batch) in [LocationScatter, LocationCovariance]:
            batch = [batch]
        n = torch.tensor(len(batch), dtype=torch.float)
        if weights is None:
            weights = torch.ones(len(batch))/n
        cost = 0
        for other, w in zip(batch, weights):
            c = sqrtm(torch.matmul(self.scatter, torch.matmul(other.covariance , self.scatter)))
            cost += w*(torch.sum((self.location-other.location)**2) + torch.sum(self.scatter**2 + other.scatter**2) - 2*torch.trace(c))
        return cost

    def sgd_step(self, batch, weights=None, step_size=1, update_scatter=True):
        if type(batch) in [LocationScatter, LocationCovariance]:
            batch = [batch]
        n = torch.tensor(len(batch), dtype=torch.float)
        if weights is None:
            weights = torch.ones(len(batch))/n
        step_loc = 0
        step_cov = 0
        scatter = self.scatter
        inv_scatter = torch.inverse(self.scatter)

        for other, w in zip(batch, weights):
            step_loc += w*other.location
            step_cov += w*sqrtm(torch.matmul(scatter, torch.matmul(other.covariance, scatter)))
        step_cov = (1 - step_size) * self.covariance + step_size * step_cov
        step_cov = torch.matmul(step_cov, step_cov)
        self.location = (1 - step_size)*self.location + step_size * step_loc
        self.covariance = torch.matmul(inv_scatter, torch.matmul(step_cov, inv_scatter))


class Location(TpModule):
    pass


class Constant(Location):
    def __init__(self, c=None, *args, **kwargs):
        super(Constant, self).__init__(*args, **kwargs)
        self.c = c

    def forward(self, x):
        return self.c


class Linear(Location):
    def __init__(self, weight=None, offset=None, *args, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)
        self.weight = weight
        self.offset = offset

    def forward(self, x):
        return self.offset+torch.mul(self.weight, x)


class Kernel(TpModule):
    def __init__(self, *args, **kwargs):
        super(Kernel, self).__init__(*args, **kwargs)

    def k(self, x1, x2):
        pass

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        return self.k(x1.view(-1, 1), x2.view(1, -1))


class WN(Kernel):
    def __init__(self, var=None, *args, **kwargs):
        super(WN, self).__init__(*args, **kwargs)
        self.var = var

    def k(self, x1, x2):
        return self.var * x1.eq(x2).float()


class BW(Kernel):
    def __init__(self, var=None, *args, **kwargs):
        super(BW, self).__init__(*args, **kwargs)
        self.var = var

    def k(self, x1, x2):
        return self.var * torch.min(x1,x2)


class SE(Kernel):
    def __init__(self, var=None, scale=None, *args, **kwargs):
        super(SE, self).__init__(*args, **kwargs)
        self.var = var
        self.scale = scale

    def k(self, x1, x2):
        return self.var * torch.exp(-((x1 - x2) ** 2) / (self.scale ** 2))


class SIN(Kernel):
    def __init__(self, var=None, scale=None, period=None, *args, **kwargs):
        super(SIN, self).__init__(*args, **kwargs)
        self.var = var
        self.scale = scale
        self.period = period

    def k(self, x1, x2):
        return self.var * torch.exp(-(torch.sin((x1 - x2) / self.period) ** 2) / (self.scale ** 2))


class COS(Kernel):
    def __init__(self, var=None, period=None, *args, **kwargs):
        super(COS, self).__init__(*args, **kwargs)
        self.var = var
        self.period = period

    def k(self, x1, x2):
        return self.var * torch.cos((x1 - x2) / self.period)
