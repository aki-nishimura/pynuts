import numpy as np


"""
Defines a (numerical) Hamiltonian dynamics based on a Gaussian momentum and the 
velocity Verlet integrator. The code is written so that other integrators & 
momentum distributions can also be employed straightwardly.
"""

class HamiltonianDynamics():

    def __init__(self):
        self.integrator = velocity_verlet
        self.momentum = GaussianMomentum()

    def integrate(self, f, dt, q, p, grad):
        q, p, logp, grad \
            = velocity_verlet(f, self.momentum.get_grad, dt, q, p, grad)
        return q, p, logp, grad

    def draw_momentum(self, n_param):
        return self.momentum.draw_random(n_param)

    def compute_hamiltonian(self, logp, p):
        potential = - logp
        kinetic = - self.momentum.get_logp(p)
        return potential + kinetic


def velocity_verlet(
        get_position_logp_and_grad, get_momentum_grad, dt, q, p, position_grad
    ):
    p = p + 0.5 * dt * position_grad
    q = q - dt * get_momentum_grad(p)
    position_logp, position_grad = get_position_logp_and_grad(q)
    p = p + 0.5 * dt * position_grad
    return q, p, position_logp, position_grad


class GaussianMomentum():

    def __init__(self):
        pass

    def draw_random(self, n_param):
        return np.random.randn(n_param)

    def get_grad(self, p):
        return - p

    def get_logp(self, p):
        return - 0.5 * np.dot(p, p)
