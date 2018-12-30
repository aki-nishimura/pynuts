import numpy as np


"""
Defines the velocity Verlet integrator for Hamiltonian dynamics based on a 
Gaussian momentum. The functions can be replaced by those for other integrators 
& momentum distributions.
"""

def integrator(f, dt, theta, p, grad):

    p = p + 0.5 * dt * grad
    theta = theta + dt * p
    logp, grad = f(theta)
    p = p + 0.5 * dt * grad

    return theta, p, logp, grad


def compute_hamiltonian(logp, p):
    return - logp + 0.5 * np.dot(p, p)


def draw_momentum(n_param):
    return np.random.randn(n_param)