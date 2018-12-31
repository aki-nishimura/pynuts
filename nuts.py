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

        # Double the size of the tree.
        if (dir == -1):
            qminus, pminus, gradminus, _, _, _, qprime, gradprime, logpprime, nprime, stopprime, alpha, nalpha, nfevals = \
                build_tree(qminus, pminus, gradminus, logu, dir, depth, dt, f, joint)
        else:
            _, _, _, qplus, pplus, gradplus, qprime, gradprime, logpprime, nprime, stopprime, alpha, nalpha, nfevals = \
                build_tree(qplus, pplus, gradplus, logu, dir, depth, dt, f, joint)
        nfevals_total += nfevals

        # Use Metropolis-Hastings to decide whether or not to move to a
        # point from the half-tree we just generated.
        if (not stopprime) and (np.random.uniform() < nprime / n):
            q = qprime
            logp = logpprime
            grad = gradprime
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

    alpha_ave = alpha / nalpha

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


def build_tree(q, p, grad, logu, dir, depth, dt, f, joint0):
    """The main recursion."""

    nfevals_total = 0
    if (depth == 0):
        # Base case: Take a single leapfrog step in the direction dir.
        qprime, pprime, logpprime, gradprime = integrator(f, dir * dt, q, p, grad)
        nfevals_total += 1
        if math.isinf(logpprime):
            joint = - float('inf')
        else:
            joint = - compute_hamiltonian(logpprime, pprime)
        # Is the new point in the slice?
        nprime = int(logu < joint)
        # Is the simulation wildly inaccurate?
        stopprime = (logu - 100) > joint
        # Set the return values---minus=plus for all things here, since the
        # "tree" is of depth 0.
        qminus = qprime
        qplus = qprime
        pminus = pprime
        pplus = pprime
        gradminus = gradprime
        gradplus = gradprime
        # Compute the acceptance probability.
        alphaprime = min(1, np.exp(joint - joint0))
        nalphaprime = 1
    else:
        # Recursion: Implicitly build the height depth-1 left and right subtrees.
        qminus, pminus, gradminus, qplus, pplus, gradplus, qprime, gradprime, logpprime, nprime, stopprime, alphaprime, nalphaprime, nfevals \
             = build_tree(q, p, grad, logu, dir, depth - 1, dt, f, joint0)
        nfevals_total += nfevals
        # No need to keep going if the stopping criteria were met in the first subtree.
        if not stopprime:
            if (dir == -1):
                qminus, pminus, gradminus, _, _, _, qprime2, gradprime2, logpprime2, nprime2, stopprime2, alphaprime2, nalphaprime2, nfevals \
                    = build_tree(qminus, pminus, gradminus, logu, dir, depth - 1, dt, f, joint0)
            else:
                _, _, _, qplus, pplus, gradplus, qprime2, gradprime2, logpprime2, nprime2, stopprime2, alphaprime2, nalphaprime2, nfevals \
                    = build_tree(qplus, pplus, gradplus, logu, dir, depth - 1, dt, f, joint0)
            nfevals_total += nfevals
            # Choose which subtree to propagate a sample up from.
            if (np.random.uniform() < nprime2 / max(nprime + nprime2, 1)):
                qprime = qprime2
                gradprime = gradprime2
                logpprime = logpprime2
            # Update the number of valid points.
            nprime = nprime + nprime2
            # Update the stopping criterion.
            stopprime = (stopprime or stopprime2 or stop_criterion(qminus, qplus, pminus, pplus))
            # Update the acceptance probability statistics.
            alphaprime = alphaprime + alphaprime2
            nalphaprime = nalphaprime + nalphaprime2

    return qminus, pminus, gradminus, qplus, pplus, gradplus, \
           qprime, gradprime, logpprime, nprime, stopprime, \
           alphaprime, nalphaprime, nfevals_total


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
