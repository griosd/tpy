## Readme
TPy: Transport Processes with PyTorch

Future features
* Toy example of GPs with PyTorch
* Priors over hyperparameters
* Standard GPs as Transports with kernels/mean functions
* Given a logp: optimize, sampling and variational
* Optimal transport for learning
* Multi-output process
* Copulas and marginals of a process
* Warped GP and Student-t process as Transports
* Sparse approximations

Ideas a desarrollar para Machine Learning con Wasserstein
1. Algoritmo numerico 'estable' para aproximar la distancia W2 entre distribuciones (test con Gaussianas, misma copula)
2. Wasserstein distance entre Mixtures of Gaussian (Discreto+Continuo vs Numérico)
3. Simulador de copulas + marginales
4. Costo gaussianas (copulas+marginales)
5. Calcular costos/transportes/baricentros de distribuciones ellipiticas (Student-t) y Archimedean
6. Wasserstein distancia/transporte/baricentro para Warped Gaussian Distributions
7. Algoritmos MCMC (Metropolis, Hamiltonean, Ensemble) en Wasserstein Space
8. Algoritmos de gradiente (momentum, nesterov, estocástico) en Wasserstein Space
9. Algoritmos de Importance Sampling para el baricentro
10. Algoritmos online (streaming de datos) para calcular baricentro
11. Definir WGP como transport process y entrenar con algoritmos desarrollados
12. Extenderlo a Warped Elliptical/Archimedean Process

Ejemplos a Generar
1. logp con flat/sharp minima
2. algoritmo de Cuesta-Albertos falla con gaussianas
3. logp con infinitos minimos (GP kernel Laplace)
4. ejemplos no-Gaussianos para calcular baricentros