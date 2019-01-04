import numpy as np
import math
import time
from .dynamics import HamiltonianDynamics
from .util import warn_message_only


dynamics = HamiltonianDynamics()
integrator = dynamics.integrate
compute_hamiltonian = dynamics.compute_hamiltonian
draw_momentum = dynamics.draw_momentum


def generate_samples(
        f, q0, dt_range, n_burnin, n_sample,
        seed=None, n_update=0, adapt_stepsize=False):

    # TODO: incorporate the stepsize adaptation.
    # TODO: return additional info.

    if seed is not None:
        np.random.seed(seed)

    if np.isscalar(dt_range):
        dt_range = np.array(2 * [dt_range])

    q = q0
    if n_update > 0:
        n_per_update = math.ceil((n_burnin + n_sample) / n_update)
    else:
        n_per_update = float('inf')
    samples = np.zeros((len(q), n_sample + n_burnin))
    logp_samples = np.zeros(n_sample + n_burnin)
    accept_prob = np.zeros(n_sample + n_burnin)

    tic = time.time()
    logp, grad = f(q)
    use_averaged_stepsize = False
    for i in range(n_sample + n_burnin):
        dt = np.random.uniform(dt_range[0], dt_range[1])
        q, logp, grad, info \
            = generate_next_state(f, dt, q, logp, grad)
        if i < n_burnin and adapt_stepsize:
            pass
            # TODO: adapt stepsize.
        elif i == n_burnin - 1:
            use_averaged_stepsize = True
        samples[:, i] = q
        logp_samples[i] = logp
        if (i + 1) % n_per_update == 0:
            print('{:d} iterations have been completed.'.format(i + 1))

    toc = time.time()
    time_elapsed = toc - tic

    return samples, logp_samples, accept_prob, time_elapsed


def generate_next_state(f, dt, q, logp, grad, max_height=10):

    p = draw_momentum(len(q))
    logp_joint = - compute_hamiltonian(logp, p)
    logp_joint_threshold = logp_joint - np.random.exponential()
        # Slicing variable in the log-scale.

    tree = TrajectoryTree(f, dt, q, p, logp, grad, logp_joint, logp_joint_threshold)
    directions = 2 * (np.random.rand(max_height) < 0.5) - 1
        # Pre-allocation of random directions is unnecessary, but makes the code easier to test.
    tree, final_height, last_doubling_rejected \
        = _grow_trajectory_recursively(tree, directions)

    info = {
        'ave_accept_prob': float('nan'),
        'ave_hamiltonian_error': float('nan'),
        'tree_height': final_height,
        'u_turn_detected': tree.u_turn_detected,
        'instability_detected': tree.instability_detected,
        'last_doubling_rejected': last_doubling_rejected
    }

    q, logp, grad = tree.sample
    return q, logp, grad, info


def _grow_trajectory_recursively(tree, directions):

    height = 0 # Referred to as 'depth' in the original paper, but arguably the
               # trajectory tree is built 'upward' on top of the existing ones.
    max_height = len(directions)
    trajectory_terminated = False
    while not trajectory_terminated:

        doubling_rejected \
            = tree.double_trajectory(height, directions[height])
            # No transition to the next half of trajectory takes place if the
            # termination criteria are met within the next half tree.

        trajectory_terminated \
            = tree.u_turn_detected or tree.instability_detected
        height += 1
        if height >= max_height and (not trajectory_terminated):
            warn_message_only(
                'The trajectory tree reached the max height of {:d} before '
                'meeting the U-turn condition.'.format(max_height)
            )
            trajectory_terminated = True

    return tree, height, doubling_rejected


class TrajectoryTree():
    """
    Collection of (a subset of) states along the simulated Hamiltonian dynamics
    trajcetory endowed with a binary tree structure.
    """

    def __init__(self, f, dt, q, p, logp, grad, joint_logp,
                 joint_logp_threshold, hamiltonian_error_tol=100):

        self.f = f
        self.dt = dt
        self.joint_logp_threshold = joint_logp_threshold
        self.front_state = (q, p, grad)
        self.rear_state = (q, p, grad)
        self.sample = (q, logp, grad)
        self.u_turn_detected = False
        self.min_hamiltonian = - joint_logp
        self.max_hamiltonian = - joint_logp
        self.hamiltonian_error_tol = hamiltonian_error_tol
        self.n_acceptable_states = int(joint_logp > joint_logp_threshold)
        self.n_integration_steps = 0

    @property
    def instability_detected(self):
        fluctuation_along_trajectory = self.max_hamiltonian - self.min_hamiltonian
        return fluctuation_along_trajectory > self.hamiltonian_error_tol

    def double_trajectory(self, height, direction):
        next_tree = self._build_next_tree(
            *self._get_states(direction), height, direction
        )
        no_transition_to_next_tree_attempted \
            = self._merge_next_tree(next_tree, direction, sampling_method='swap')
        return no_transition_to_next_tree_attempted

    def _build_next_tree(self, q, p, grad, height, direction):

        if height == 0:
            return self._build_next_singleton_tree(q, p, grad, direction)

        subtree = self._build_next_tree(q, p, grad, height - 1, direction)
        trajectory_terminated_within_subtree \
            = subtree.u_turn_detected or subtree.instability_detected
        if not trajectory_terminated_within_subtree:
            next_subtree = self._build_next_tree(
                *subtree._get_states(direction), height - 1, direction
            )
            subtree._merge_next_tree(next_subtree, direction, sampling_method='uniform')

        return subtree

    def _build_next_singleton_tree(self, q, p, grad, direction):
        q, p, logp, grad = integrator(self.f, direction * self.dt, q, p, grad)
        self.n_integration_steps += 1
        if math.isinf(logp):
            joint_logp = - float('inf')
        else:
            joint_logp = - compute_hamiltonian(logp, p)
        return TrajectoryTree(
            self.f, self.dt, q, p, logp, grad, joint_logp, self.joint_logp_threshold
        )

    def _merge_next_tree(self, next_tree, direction, sampling_method):

        self.u_turn_detected = self.u_turn_detected or next_tree.u_turn_detected
        self.min_hamiltonian = min(self.min_hamiltonian, next_tree.min_hamiltonian)
        self.max_hamiltonian = max(self.max_hamiltonian, next_tree.max_hamiltonian)
        trajectory_terminated_within_next_tree \
            = next_tree.u_turn_detected or next_tree.instability_detected

        if not trajectory_terminated_within_next_tree:
            self._update_sample(next_tree, sampling_method)
            self.n_acceptable_states += next_tree.n_acceptable_states
            self._set_states(*next_tree._get_states(direction), direction)
            self.u_turn_detected \
                = self.u_turn_detected or self._check_u_turn_at_front_and_rear_ends()

        return trajectory_terminated_within_next_tree

    def _update_sample(self, next_tree, method):
        """
        Parameters
        ----------
        method: {'uniform', 'swap'}
        """
        if method == 'uniform':
            n_total = self.n_acceptable_states + next_tree.n_acceptable_states
            sampling_weight_on_next_tree \
                = next_tree.n_acceptable_states / max(1, n_total)
        elif method == 'swap':
            sampling_weight_on_next_tree \
                = next_tree.n_acceptable_states / self.n_acceptable_states
        if np.random.uniform() < sampling_weight_on_next_tree:
            self.sample = next_tree.sample

    def _check_u_turn_at_front_and_rear_ends(self):
        q_front, p_front, _ = self._get_states(1)
        q_rear, p_rear, _ = self._get_states(-1)
        dq = q_front - q_rear
        return (np.dot(dq, p_front) < 0) or (np.dot(dq, p_rear) < 0)

    def _set_states(self, q, p, grad, direction):
        if direction > 0:
            self.front_state = (q, p, grad)
        else:
            self.rear_state = (q, p, grad)

    def _get_states(self, direction):
        if direction > 0:
            return self.front_state
        else:
            return self.rear_state


# TODO: replace the following functions with 'generate_samples' function above.

def nuts_adap(f, n_sample, n_warmup, q0, delta=0.8, seed=None, dt=None, n_update=10):
    """
    Implements the No-U-Turn Sampler (NUTS) of Hoffman & Gelman, 2011.
    Runs n_warmup steps of burn-in, during which it adapts the step size
    parameter dt, then starts generating samples to return.
    INPUTS
    ------
    dt: float
        step size
    f: callable
        it should return the log probability and gradient evaluated at q
        logp, grad = f(q)
    n_mcmc: int
        number of samples to generate.
    n_warmup: int
        the number of steps of burn-in/how long to run the dual averaging
        algorithm to fit the step size dt.
    q0: ndarray[float, ndim=1]
        initial guess of the parameters.
    KEYWORDS
    --------
    delta: float
        targeted average acceptance rate
    OUTPUTS
    -------
    samples: ndarray[float, ndim=2]
    """

    np.random.seed(seed)

    if len(np.shape(q0)) > 1:
        raise ValueError('q0 is expected to be a 1-D array')

    if n_warmup > 0:
        q, dt, _, _ = dualAveraging(f, q0, delta, n_warmup)
        print('The warmup iterations have been completed.')
    else:
        q = q0.copy()

    n_per_update = math.ceil(n_sample / n_update)

    nfevals_total = 0
    samples = np.zeros((n_sample, len(q)))
    logp_samples = np.zeros(n_sample)
    accept_prob = np.zeros(n_sample)

    tic = time.process_time()
    logp, grad = f(q)
    for i in range(n_sample):
        q, logp, grad, alpha_ave, nfevals = generate_next_state(f, dt, q, logp, grad)
        nfevals_total += nfevals
        samples[i,:] = q
        logp_samples[i] = logp
        accept_prob[i] = alpha_ave
        if (i + 1) % n_per_update == 0:
            print('{:d} iterations have been completed.'.format(i+1))
    toc = time.process_time()
    time_elapsed = toc - tic
    nfevals_per_itr = nfevals_total / n_sample

    return samples, logp_samples, dt, accept_prob, nfevals_per_itr, time_elapsed


def dualAveraging(f, q0, delta=.8, n_warmup=500):

    logp, grad = f(q0)
    if math.isinf(logp):
        raise ValueError('The log density at the initial state was infinity.')

    # Choose a reasonable first dt by a simple heuristic.
    dt = find_reasonable_dt(q0, grad, logp, f)

    # Parameters to the dual averaging algorithm.
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    mu = math.log(10. * dt)

    # Initialize dual averaging algorithm.
    dtbar = 1
    dt_seq = np.zeros(n_warmup + 1)
    dtbar_seq = np.zeros(n_warmup + 1)
    dt_seq[0] = dt
    dtbar_seq[0] = dtbar

    Hbar = 0
    q = q0
    for i in range(1, n_warmup + 1):
        q, logp, grad, ave_alpha, _ = \
                generate_next_state(f, dt, q, logp, grad)
        eta = 1 / (i + t0)
        Hbar = (1 - eta) * Hbar + eta * (delta - ave_alpha)
        dt = math.exp(mu - math.sqrt(i) / gamma * Hbar)
        eta = i ** -kappa
        dtbar = math.exp((1 - eta) * math.log(dtbar) + eta * math.log(dt))
        dt_seq[i] = dt
        dtbar_seq[i] = dtbar

    return q, dtbar, dt_seq, dtbar_seq


def find_reasonable_dt(q0, grad0, logp0, f):
    """ Heuristic for choosing an initial value of dt """

    dt = 1.0
    p0 = np.random.normal(0, 1, len(q0))

    # Figure out what direction we should be moving dt.
    _, pprime, logpprime, gradprime = integrator(f, dt, q0, p0, grad0)
    if math.isinf(logpprime):
        acceptprob = 0
    else:
        acceptprob = np.exp(logpprime - logp0 - 0.5 * (np.dot(pprime, pprime) - np.dot(p0, p0)))
    a = 2 * int(acceptprob > 0.5) - 1

    # Keep moving dt in that direction until acceptprob crosses 0.5.
    while acceptprob == 0 or ((2 * acceptprob) ** a > 1):
        dt = dt * (2 ** a)
        _, pprime, logpprime, _ = integrator(f, dt, q0, p0, grad0)
        if math.isinf(logpprime):
            acceptprob = 0
            if a == 1: # The last doubling of stepsize was too much.
                dt = dt / 2
                break
        else:
            acceptprob = np.exp(logpprime - logp0 - 0.5 * (np.dot(pprime, pprime) - np.dot(p0, p0)))

    return dt
