import numpy as np
from .. import hmc
from .distributions import BivariateGaussian


# TODO: Test the use of non-identity mass matrices.

def test_trajectory_symmetry():

    f, stability_limit, n_step, q0, p0 = setup_gaussian_target()
    dt = .9 * stability_limit

    logp0, grad0 = f(q0)
    q, p, logp, info, info_reverse = simulate_forward_and_backward(
        f, dt, n_step, q0, p0, logp0, grad0, hamiltonian_tol=float('inf')
    )
    assert np.allclose(logp0, logp, atol=1e-10)
    assert np.allclose(q0, q, atol=1e-10)
    assert np.allclose(p0, p, atol=1e-10)


def test_early_termination_symmetry():

    f, stability_limit, n_step, q0, p0 = setup_gaussian_target()
    dt = .99 * stability_limit

    hamiltonian_fluctuation_range \
        = find_max_hamiltonian_fluctuation(f, dt, n_step, q0, p0)

    # Expect early termination.
    hamiltonian_tol = hamiltonian_fluctuation_range - .05
    logp0, grad0 = f(q0)
    q, p, logp, info, info_reverse = simulate_forward_and_backward(
        f, dt, n_step, q0, p0, logp0, grad0, hamiltonian_tol
    )
    assert info['instability_detected'] == info_reverse['instability_detected'] == True

    # Expect NO early termination.
    hamiltonian_tol = hamiltonian_fluctuation_range + .05
    logp0, grad0 = f(q0)
    q, p, logp, info, info_reverse = simulate_forward_and_backward(
        f, dt, n_step, q0, p0, logp0, grad0, hamiltonian_tol
    )
    assert info['instability_detected'] == info_reverse['instability_detected'] == False


def setup_gaussian_target():

    bi_gauss = BivariateGaussian(rho=.9, sigma=np.array([1., 2.]))
    f = bi_gauss.compute_logp_and_gradient
    stability_limit = bi_gauss.get_stepsize_stability_limit()
    n_step = 100
    q0 = np.array([0., 0.])
    p0 = np.array([1., 1.])

    return f, stability_limit, n_step, q0, p0


def simulate_forward_and_backward(f, dt, n_step, q0, p0, logp0, grad0,
                                  hamiltonian_tol):
    # Forward dynamics.
    q, p, logp, grad, info = hmc.simulate_dynamics(
        f, dt, n_step, q0, p0, logp0, grad0, hamiltonian_tol
    )

    # Reverse dynamics.
    q, p, logp, grad, reverse_info = hmc.simulate_dynamics(
        f, dt, n_step, q, -p, logp, grad, hamiltonian_tol
    )
    p = -p

    return q, p, logp, info, reverse_info


def find_max_hamiltonian_fluctuation(f, dt, n_step, q0, p0):

    logp0, grad0 = f(q0)
    _, _, _, _, info = hmc.simulate_dynamics(
        f, dt, n_step, q0, p0, logp0, grad0, hamiltonian_tol=float('inf')
    )
    return np.ptp(info['energy_trajectory'])