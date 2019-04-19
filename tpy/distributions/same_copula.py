from .. import TransportFamily


class SameCopula(TransportFamily):
    def __init__(self, copula, marginals=None):
        self.copula = copula
        self.marginals = marginals
