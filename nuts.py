import numpy as np
import math
import time
from .dynamics import HamiltonianDynamics


dynamics = HamiltonianDynamics()
integrator = dynamics.integrate
compute_hamiltonian = dynamics.compute_hamiltonian
random_momentum = dynamics.draw_momentum


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
        q, logp, grad, alpha_ave, nfevals_total \
            = nuts(f, dt, q, logp, grad)
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


def nuts(f, dt, q, logp, grad, max_depth=10, warnings=True):

    d = len(q)

    # Resample momenta.
    p = random_momentum(d)

    # joint lnp of q and momentum r
    joint = - compute_hamiltonian(logp, p)

    # Resample u ~ uniform([0, exp(joint)]).
    # Equivalent to (log(u) - joint) ~ exponential(1).
    logu = joint - np.random.exponential()

    # initialize the tree
    tree = TrajectoryTree(q, p, logp, grad, joint, logu)
    qminus = q
    qplus = q
    pminus = p
    pplus = p
    gradminus = grad
    gradplus = grad

    nfevals_total = 0
    depth = 0
    n = 1  # Initially the only valid point is the initial point.
    stop = False
    while not stop:
        # Choose a direction. -1 = backwards, 1 = forwards.
        dir = int(2 * (np.random.uniform() < 0.5) - 1)

        next_tree = build_next_tree(f, dt, *tree.get_states(dir), depth, dir, logu)
        nprime = next_tree.n_acceptable_states
        stopprime = next_tree.u_turn_detected or next_tree.trajectory_is_unstable

        if dir == -1:
            qminus, pminus, gradminus = next_tree.get_states(-1)
        else:
            qplus, pplus, gradplus = next_tree.get_states(1)

        if not stopprime:
            tree.merge_next_tree(next_tree, dir, sampling_method='swap')
            q, logp, grad = tree.get_sample()

        # Update number of valid points we've seen.
        n += nprime
        # Decide if it's time to stop.
        stop = stopprime or stop_criterion(qminus, qplus, pminus, pplus)
        # Increment depth.
        depth += 1
        if depth >= max_depth:
            stop = True
            if warnings:
                print('The max depth of {:d} has been reached.'.format(max_depth))

    # TODO: take care of the accetance probability related quantities later.
    alpha_ave = 1

    return q, logp, grad, alpha_ave, nfevals_total


def stop_criterion(qminus, qplus, pminus, pplus):
    """ Check for the U-turn condition.
    INPUTS
    ------
    qminus, qplus: ndarray[float, ndim=1]
        under and above position
    pminus, pplus: ndarray[float, ndim=1]
        under and above momentum
    """
    dq = qplus - qminus
    return (np.dot(dq, pminus) < 0) or (np.dot(dq, pplus) < 0)


def build_next_tree(f, dt, q, p, grad, height, direction, logu):

    if height == 0:
        return build_singleton_tree(f, dt, q, p, grad, direction, logu)

    subtree = build_next_tree(f, dt, q, p, grad, height - 1, direction, logu)
    if not (subtree.u_turn_detected or subtree.trajectory_is_unstable):
        q, p, grad = subtree.get_states(direction)
        next_subtree = build_next_tree(f, dt, q, p, grad, height - 1, direction, logu)
        subtree.merge_next_tree(next_subtree, direction, sampling_method='uniform')

    return subtree


def build_singleton_tree(f, dt, q, p, grad, direction, logu):
    q, p, logp, grad = integrator(f, direction * dt, q, p, grad)
    if math.isinf(logp):
        log_joint = - float('inf')
    else:
        log_joint = - compute_hamiltonian(logp, p)
    return TrajectoryTree(q, p, logp, grad, log_joint, logu)


class TrajectoryTree():
    """
    Collection of (a subset of) states along the simulated Hamiltonian dynamics
    trajcetory endowed with a binary tree structure.
    """

    def __init__(self, q0, p0, logp0, grad0, log_joint0, log_joint_threshold):
        # Store the frontmost and rearmost states of the trajectory as well as
        # one inner state sampled uniformly from the acceptable states.
        n_states_to_store = 3
        self.positions = n_states_to_store * [q0]
        self.momentums = n_states_to_store * [p0]
        self.momentums[self.get_index(direction=0)] = None
            # No use for the momentum except at the front and rear of the trajectory.
        self.gradients = n_states_to_store * [grad0]
        self.logp_at_sampling_location = logp0
        self.u_turn_detected = False
        self.trajectory_is_unstable = False
        self.n_acceptable_states = int(log_joint0 > log_joint_threshold)

    def set_states(self, q, p, grad, direction):
        index = self.get_index(direction)
        self.positions[index] = q
        self.momentums[index] = p
        self.gradients[index] = grad

    def get_states(self, direction):
        index = self.get_index(direction)
        return self.positions[index], self.momentums[index], self.gradients[index]

    def set_sample(self, q, logp, grad):
        index = self.get_index(direction=0)
        self.positions[index] = q
        self.gradients[index] = grad
        self.logp_at_sampling_location = logp

    def get_sample(self):
        index = self.get_index(direction=0)
        return self.positions[index], self.logp_at_sampling_location, self.gradients[index]

    def get_index(self, direction):
        return 1 + direction

    def merge_next_tree(self, next_tree, direction, sampling_method):
        self.set_states(*next_tree.get_states(direction), direction)
        u_turn_detected_within_subtrees \
            = self.u_turn_detected or next_tree.u_turn_detected
        self.u_turn_detected = (
            self.check_u_turn_at_front_and_rear_ends()
            or u_turn_detected_within_subtrees
        )
        self.update_sample(next_tree, sampling_method)
        self.n_acceptable_states += next_tree.n_acceptable_states

    def check_u_turn_at_front_and_rear_ends(self):
        q_front, p_front, _ = self.get_states(1)
        q_rear, p_rear, _ = self.get_states(-1)
        dq = q_front - q_rear
        return (np.dot(dq, p_front) < 0) or (np.dot(dq, p_rear) < 0)

    def update_sample(self, next_tree, method):
        """
        Parameters
        ----------
        method: {'uniform', 'swap'}
        """
        if next_tree.u_turn_detected or next_tree.trajectory_is_unstable:
            np.random.uniform()
            return

        if method == 'uniform':
            n_total = self.n_acceptable_states + next_tree.n_acceptable_states
            sampling_weight_on_next_tree \
                = next_tree.n_acceptable_states / max(1, n_total)
        elif method == 'swap':
            sampling_weight_on_next_tree \
                = next_tree.n_acceptable_states / self.n_acceptable_states
        if np.random.uniform() < sampling_weight_on_next_tree:
            self.set_sample(*next_tree.get_sample())


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
        q, logp, grad, alpha_ave, nfevals = nuts(f, dt, q, logp, grad)
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
                nuts(f, dt, q, logp, grad)
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
