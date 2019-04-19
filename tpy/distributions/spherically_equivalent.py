import torch
from .. import TransportFamily


class SphericallyEquivalent(TransportFamily):
    def __init__(self, generator, radial=None):
        self.generator = generator
        self.radial = radial

    def sample(self, n_samples=None):
        s = self.generator.sample(n_samples)
        norm = torch.norm(s)
        return (self.radial(norm)/norm)*s

    def log_prob(self, r):
        rnorm = torch.norm(r)
        xnorm = self.radial.inv(rnorm)
        x = (xnorm/rnorm)*r
        return self.generator.log_prob(x) # + logdet

    def w2(self, other):
        pass

    def ot(self, other):
        self.radial