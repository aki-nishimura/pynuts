import numpy as np


class BivariateGaussian():

    def __init__(self, rho=.9, sigma=np.array([1.0, 2.0])):
        Sigma = np.array([
            [sigma[0] ** 2, rho * np.prod(sigma)],
            [rho * np.prod(sigma), sigma[1] ** 2]
        ])
        self.sigma = sigma
        self.Sigma = Sigma
        self.Phi = np.linalg.inv(Sigma)

    def compute_logp_and_gradient(self, x, logp_only=False):
        grad = - self.Phi.dot(x)
        logp = np.inner(x, grad) / 2
        return logp, grad

    def compute_marginal_pdf(self, x_1, x_2):
        x = [x_1, x_2]
        pdf = [
            1 / np.sqrt(2 * np.pi) / self.sigma[axis] \
                * np.exp(- x[axis] ** 2 / 2 / self.sigma[axis] ** 2)
            for axis in [0, 1]
        ]
        return pdf

    def get_principal_components(self):
        """ Returns principal component variances and directions. """
        eigvals, eigvec = np.linalg.eig(self.Phi)
        return 1 / eigvals, eigvec

    def get_stepsize_stability_limit(self):
        pc_scale = np.sqrt(
            self.get_principal_components()[0]
        )
        return 2 * np.min(pc_scale)


class BivariateBanana():
    """
    Define a posterior under the model
        y | x_1, x_2 \sim Normal(mean = x_1 + x_2^2, sigma^2)
        x_1, x_2 \sim Normal(0, 1)
    """

    def __init__(self, y=1., sigma=.5):
        self.y = y
        self.phi = sigma ** -2

    def compute_logp_and_gradient(self, x, logp_only=False):
        logp = - np.sum(x ** 2) / 2
        logp += - self.phi / 2 * (self.y - x[0] - x[1] ** 2) ** 2
        grad = - x
        grad += self.phi * (self.y - x[0] - x[1] ** 2) \
            * np.array([1, 2 * x[1]])
        return logp, grad

    def compute_marginal_pdf(self, x_1, x_2):
        logp = np.zeros((len(x_1), len(x_2)))
        for i in range(len(x_1)):
            for j in range(len(x_2)):
                logp[i, j] = self.compute_logp_and_gradient(np.array([x_1[i], x_2[j]]))[0]

        joint_pdf = np.exp(logp)
        pdf_1 = np.trapz(joint_pdf, x_2, axis=-1)
        pdf_1 /= np.trapz(pdf_1, x_1)
        pdf_2 = np.trapz(joint_pdf, x_1, axis=0)
        pdf_2 /= np.trapz(pdf_2, x_2, axis=0)
        return pdf_1, pdf_2
    