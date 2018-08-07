import numpy as np
from scipy.stats import norm

class Copula:
    pass

class GaussianCopula(Copula):
    def __init__(self, loc=0, scale=1, corr=0, nsamples=1):
        self.loc = loc
        self.scale = scale
        self.corr = corr
        R = np.array([[1, self.corr], [self.corr, 1]])*self.scale**2
        self.samples = np.random.multivariate_normal(np.ones(2)*self.loc, R, nsamples)
        
    def _sample(nsamples = 1):
        R = np.array([[1, self.corr], [self.corr, 1]])*self.scale**2
        self.samples = np.random.multivariate_normal(np.ones(2)*self.loc, R, nsamples)
    
    def joint(self):
        return self.samples
    
    def marginals(self):
        return self.samples[:, 0], self.samples[:, 1]
    
    def copula(self):
        C1 = norm.cdf(self.samples[:, 0], loc=self.loc, scale=self.scale)
        C2 = norm.cdf(self.samples[:, 1], loc=self.loc, scale=self.scale)
        return np.vstack([C1, C2]).T
        
        
class Archimedean(Copula):
    pass


class Ali_Mikhail_Haq(Archimedean):
    """theta in [-1,1)"""
    theta_u = 0.0
    theta_v = 0.999

    def phi_1(self, t, theta=0.5):
        return np.log((1-theta*(1-t))/t)

    def phi(self, t, theta=0.5):
        return (1-theta)/(np.exp(t)-theta)


class Clayton(Archimedean):
    """theta in [-1,inf)\{0}"""
    theta_u = 0.0001
    theta_v = 4

    def phi_1(self, t, theta=0.5):
        return (t**(-theta)-1)/theta

    def phi(self, t, theta=0.5):
        return (1+theta*t)**(-1/theta)


class Gumbel(Archimedean):
    """theta in [1,inf)"""
    theta_u = 1.0
    theta_v = 4

    def phi_1(self, t, theta=0.5):
        return (-np.log(t)) ** theta

    def phi(self, t, theta=0.5):
        return np.exp(-t ** (1 / theta))
