import numpy as np


def privacy_lower_bounds_vector(C, epsilon, delta, gamma):
    norm_2 = np.linalg.norm(C, ord=2)
    norm_1 = np.linalg.norm(C, ord=1)
    sigma2_lb = 2.0 * np.log(1.25 / delta) * (norm_2**2) * (gamma**2) / (epsilon**2)
    b_lb = norm_1 * gamma / epsilon
    return float(sigma2_lb), float(b_lb)


def kl_radius_vector(sigma2_nom, sigma2_ub, b_lb, b_ub, L):
    def gfun(x):
        return np.log(sigma2_nom / x) + x / sigma2_nom

    eta1 = float(gfun(sigma2_ub) - 1.0)
    eta2 = float(max(gfun(2.0 * b_lb**2), gfun(2.0 * b_ub**2)) - 2.0 + np.log(np.pi))
    eta = 0.5 * L * max(eta1, eta2)
    return float(eta), eta1, eta2
