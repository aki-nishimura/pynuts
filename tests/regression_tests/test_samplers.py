import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')

import numpy as np
from hamiltonian_monte_carlo import hmc
from hamiltonian_monte_carlo.nuts import NoUTurnSampler
from hamiltonian_monte_carlo.tests.distributions import BivariateGaussian

data_folder = 'saved_outputs'


def test_hmc(request):
    samples = run_sampler('hmc')
    filepath = get_file_path(request.fspath.dirname, 'hmc')
    prev_output = np.load(filepath)
    assert np.allclose(samples[:, -1], prev_output, atol=1e-10, rtol=1e-10)

def test_nuts(request):
    samples = run_sampler('nuts')
    filepath = get_file_path(request.fspath.dirname, 'nuts')
    prev_output = np.load(filepath)
    assert np.allclose(samples[:, -1], prev_output, atol=1e-10, rtol=1e-10)


def run_sampler(sampler):
    # sampler : {'hmc', 'nuts'}

    dt = np.array([.55, .63])
    n_step = np.array([3, 4])
    n_burnin = 10
    n_sample = 100
    seed = 0

    theta0 = np.zeros(2)
    f = BivariateGaussian().compute_logp_and_gradient

    if sampler == 'hmc':
        samples = hmc.generate_samples(
            f, theta0, n_burnin, n_sample, n_step, dt, seed=seed)[0]
    elif sampler == 'nuts':
        nuts = NoUTurnSampler(f)
        samples = nuts.generate_samples(
            theta0, n_burnin, n_sample, dt, seed=seed)[0]
    else:
        raise NotImplementedError()

    return samples


def get_file_path(tests_dirname, sampler):
    filepath = '/'.join([
        tests_dirname, data_folder,
        sampler + '_bivariate_gaussian_samples.npy'
    ])
    return filepath


# Update the saved HMC outputs, if called as a script with option 'update'.
if __name__ == '__main__':
    option = sys.argv[-1]
    if option == 'update':
        for sampler in ['hmc', 'nuts']:
            samples = run_sampler(sampler)
            filepath = get_file_path('.', sampler)
        np.save(filepath, samples[:, -1])