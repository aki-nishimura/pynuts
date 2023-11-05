import numpy as np
import math


"""
Defines a (numerical) Hamiltonian dynamics based on a Gaussian momentum and the 
velocity Verlet integrator. The code is written so that other integrators & 
momentum distributions can also be employed straightforwardly.
"""

class HamiltonianDynamics():

    def __init__(self, mass=None, momentum_dist='gaussian'):
        """
        Parameters
        ----------
        mass: None, numpy 1d array, or callable `mass(p, power)`
            If callable, should return a vector obtained by multiplying the
            vector p with matrix M ** power for power \in {-1, 1/2} for Gaussian
            and power \in {-1/2, 1/2} for Laplace momentum, where M represents
            a mass matrix defined as the covariance of momentum.
            The matrix L corresponding to M ** 1/2 only needs to satisfy L L' = M.
            Passing M = None defaults to a dynamics with the identity mass matrix.
        """

        if mass is None:
            mass_operator = lambda p, power: p
        elif isinstance(mass, np.ndarray):
            sqrt_mass = np.sqrt(mass)
            inv_mass = 1 / mass
            def mass_operator(p, power):
                if power == -1:
                    return inv_mass * p
                elif power == - 1 / 2:
                    return p / sqrt_mass
                elif power == 1 / 2:
                    return sqrt_mass * p
        elif callable(mass):
            if momentum_dist == 'gaussian':
                raise ValueError(
                    "Callable mass matrix for non-Gaussian momentum is unsupported."
                )
            mass_operator = mass
        else:
            raise ValueError("Unsupported type for the mass matrix.")

        if momentum_dist == 'gaussian':
            self.integrator = velocity_verlet
            self.momentum = GaussianMomentum(mass_operator)
        elif momentum_dist == 'laplace':
            self.integrator = midpoint
            self.momentum = LaplaceMomentum(mass_operator)
        else:
            raise ValueError("Requested momentum distribution is unsupported.")


    def integrate(self, get_logp_and_grad, dt, q, p, grad):
        """
        Parameters
        ----------
        get_logp_and_grad: callable
            Funtion to return the log density (not potential energy, so watch the
            sign) and its gradient.
        """
        q, p, logp, grad \
            = self.integrator(
                get_logp_and_grad, self.momentum.get_grad, dt, q, p, grad
            )
        return q, p, logp, grad

    def draw_momentum(self, n_param):
        return self.momentum.draw_random(n_param)

    def compute_hamiltonian(self, logp, p):
        potential = - logp
        kinetic = - self.momentum.get_logp(p)
        return potential + kinetic

    def convert_to_velocity(self, p):
        return - self.momentum.get_grad(p)


def velocity_verlet(
        get_position_logp_and_grad, get_momentum_grad, dt, q, p, position_grad
    ):
    p = p + 0.5 * dt * position_grad
    q = q - dt * get_momentum_grad(p)
    position_logp, position_grad = get_position_logp_and_grad(q)
    if math.isfinite(position_logp):
        p += 0.5 * dt * position_grad
    return q, p, position_logp, position_grad


def midpoint(
        get_position_logp_and_grad, get_momentum_grad, dt, q, p, position_grad
    ):
    # Unused parameter `position_grad` acts only as a placeholder to make the
    # number of arguments consistent with other integrators.
    q = q - 0.5 * dt * get_momentum_grad(p)
    position_logp, position_grad = get_position_logp_and_grad(q)
    if math.isfinite(position_logp):
        p_prop = p + dt * position_grad
        vel_changed = (np.sign(p_prop) != np.sign(p))
        p[vel_changed] = - p[vel_changed]
        p[np.logical_not(vel_changed)] = p_prop[np.logical_not(vel_changed)]
        q = q - 0.5 * dt * get_momentum_grad(p)
        position_logp, _ = get_position_logp_and_grad(q)
    position_grad = None # Cannot recycle the gradient at next integration step
    return q, p, position_logp, position_grad


class GaussianMomentum():

    def __init__(self, mass=None):
        self.mass = mass

    def draw_random(self, n_param):
        p = self.mass(np.random.randn(n_param), 1/2)
        return p

    def get_grad(self, p):
        return - self.mass(p, -1)

    def get_logp(self, p):
        return - 0.5 * np.dot(p, self.mass(p, -1))

class LaplaceMomentum():

    def __init__(self, mass=None):
        # Mass matrix here is defined as the inverse of the covariance of momentum
        # and differs from Nishimura et. al. (2020) which uses the scale.
        self.mass = mass

    def draw_random(self, n_param):
        p = self.mass(np.random.laplace(size=n_param), 1/2)
        return p

    def get_grad(self, p):
        return - self.mass(np.sign(p), -1/2)

    def get_logp(self, p):
        return - np.sum(self.mass(np.abs(p), -1/2))