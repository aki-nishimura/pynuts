import numpy as np
from .. import nuts
from .distributions import BivariateGaussian, BivariateBanana


# TODO: Test the use of non-identity mass matrices.

def test_tree_building_algorithm():

    f = BivariateBanana().compute_logp_and_gradient
    q_1 = np.array([1., 0.])
    p_1 = np.array([0., 1.])
    logp_1, grad_1 = f(q_1)
    dt = .01 # Take a small enough stepsize that the U-turn condition does not occur for a while.

    # Doule tree backward for a while followed by one forward doubling.
    directions = np.array([-1, -1, -1, 1])
    tree_1, _, _ = simulate_nuts_tree_dynamics(
        f, dt, q_1, p_1, logp_1, grad_1, directions
    )
    # Try generating the same tree from a neighboring state
    q_2, p_2, logp_2, grad_2 = nuts.integrator(f, dt, q_1, p_1, grad_1)
    directions = - directions
    tree_2, _, _ = simulate_nuts_tree_dynamics(
        f, dt, q_2, p_2, logp_2, grad_2, directions
    )
    assert has_same_front_and_rear_states(tree_1, tree_2)

    # Run another check.
    directions = np.array([1, 1, -1])
    tree_1, _, _ = simulate_nuts_tree_dynamics(
        f, dt, q_1, p_1, logp_1, grad_1, directions
    )
    directions = np.array([-1, 1, -1])
    tree_2, _, _ = simulate_nuts_tree_dynamics(
        f, dt, q_2, p_2, logp_2, grad_2, directions
    )
    assert has_same_front_and_rear_states(tree_1, tree_2)


def test_u_turn():

    bi_gauss = BivariateGaussian(rho=.0, sigma=np.array([1., .1]))
    f = bi_gauss.compute_logp_and_gradient

    # Set up the values for the tree trajectory so that the U-turn condition
    # occurs exactly at the last tree doubling.
    q_1 = np.array([-1., 0.])
    p_1 = np.array([0.01, 0.])
    logp_1, grad_1 = f(q_1)
    tree_height = 10
    dt = np.pi / 2 ** (tree_height - 1)
        # Pick dt so that the trajectory is perioric with 2 ** tree_height steps-ish.
    dt *= .9 # Make sure the U-turn occurs around, but not earlier than, 2 ** (tree_height - 1) steps.

    # Doule tree forward till the U-turn almost occurs.
    directions_1 = np.ones(tree_height - 1)
    tree_1, final_height_1, last_doubling_rejected_1 \
        = simulate_nuts_tree_dynamics(f, dt, q_1, p_1, logp_1, grad_1, directions_1)
    assert last_doubling_rejected_1 == tree_1.u_turn_detected == False

    # Add one final doubling backward: now the U-turn should occur.
    directions_1 = np.concatenate((directions_1, [-1]))
    tree_1, final_height_1, last_doubling_rejected_1 \
        = simulate_nuts_tree_dynamics(f, dt, q_1, p_1, logp_1, grad_1, directions_1)

    # Simulate the same tree from the frontmost node of the previous tree.
    directions_2 = - np.ones(tree_height)
    q_2, p_2, _ = tree_1.front_state
    logp_2, grad_2 = f(q_2)
    tree_2, final_height_2, last_doubling_rejected_2 \
        = simulate_nuts_tree_dynamics(f, dt, q_2, p_2, logp_2, grad_2, directions_2)

    assert has_same_front_and_rear_states(tree_1, tree_2)
    assert final_height_1 == final_height_2 == tree_height
    assert last_doubling_rejected_1 == last_doubling_rejected_2 == True


def has_same_front_and_rear_states(tree_1, tree_2):

    q_1, p_1, _ = tree_1.front_state
    q_2, p_2, _ = tree_2.front_state
    result = np.allclose(q_1, q_2) and np.allclose(p_1, p_2)

    q_1, p_1, _ = tree_1.rear_state
    q_2, p_2, _ = tree_2.rear_state
    result = result and np.allclose(q_1, q_2) and np.allclose(p_1, p_2)

    return result


def setup_gaussian_target():
    bi_gauss = BivariateGaussian(rho=.9, sigma=np.array([1., 2.]))
    f = bi_gauss.compute_logp_and_gradient
    stability_limit = bi_gauss.get_stepsize_stability_limit()
    q0 = np.array([0., 0.])
    p0 = np.array([1., 1.])
    logp0, grad0 = f(q0)

    return f, stability_limit, q0, p0, logp0, grad0


def simulate_nuts_tree_dynamics(
        f, dt, q0, p0, logp0, grad0, directions,
        hamiltonian_error_tol=float('inf')):

    logp_joint = - nuts.compute_hamiltonian(logp0, p0)
    logp_joint_threshold = - float('inf')
        # Enforce all the states along the trajectory to be acceptable.

    tree = nuts.TrajectoryTree(
        f, dt, q0, p0, logp0, grad0, logp_joint, logp_joint_threshold,
        hamiltonian_error_tol=hamiltonian_error_tol
    )
    tree, final_height, last_doubling_rejected \
        = nuts._grow_trajectory_recursively(tree, directions)

    return tree, final_height, last_doubling_rejected
