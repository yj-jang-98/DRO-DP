import math

import numpy as np
import pandas as pd


def simulate_controller_vector_vectorized(
    controller, 
    A, B, C, Q, R, QN,
    Sigma_w, x_ini, Sigma_ini,
    noise_family,
    noise_param,
    N,
    n_mc=1000,
    seed=0,
):
    rng = np.random.default_rng(seed)
    n = A.shape[0]

    # x shape: (n_mc, n, 1)
    x = rng.multivariate_normal(x_ini.flatten(), Sigma_ini, size=n_mc)[..., np.newaxis]
    xhat = np.tile(x_ini, (n_mc, 1, 1))
    cost = np.zeros(n_mc)

    for k in range(N):
        u = -controller["F"][k] @ xhat

        # Stage cost is accumulated pathwise over the Monte Carlo batch.
        x_cost = np.squeeze(np.transpose(x, (0, 2, 1)) @ Q @ x)
        u_cost = np.squeeze(np.transpose(u, (0, 2, 1)) @ R @ u)
        cost += 0.5 * (x_cost + u_cost)

        if noise_family == "gaussian":
            v = rng.normal(0.0, math.sqrt(noise_param), size=(n_mc, 1, 1))
        elif noise_family == "laplace":
            v = rng.laplace(0.0, noise_param, size=(n_mc, 1, 1))

        # Intentional noise injection
        ytilde = C @ x + v

        if controller["kind"] == "minimax":
            tau = controller["tau"]
            # The minimax estimator adds the tau-dependent correction term.
            xhat = (
                A @ xhat
                + B @ u
                + controller["Kest"][k] @ (ytilde - C @ xhat)
                + A @ np.linalg.inv(controller["P"][k]) @ (Q / tau) @ xhat
            )
        else:
            # Standard LQG controller
            xhat = A @ xhat + B @ u + controller["Kest"][k] @ (ytilde - C @ xhat)

        w = rng.multivariate_normal(np.zeros(n), Sigma_w, size=n_mc)[..., np.newaxis]
        x = A @ x + B @ u + w

    x_term_cost = np.squeeze(np.transpose(x, (0, 2, 1)) @ QN @ x)
    cost += 0.5 * x_term_cost
    return cost


def evaluate_grid(
    controller, A, B, C, Q, R, QN, Sigma_w, x_ini, Sigma_ini, noise_family, param_grid, N, n_mc=1000, seed=0
):
    rows = []
    for i, param in enumerate(param_grid):
        # Shift the seed across grid points so each parameter value gets an independent batch.
        costs = simulate_controller_vector_vectorized(
            controller=controller,
            A=A, B=B, C=C, Q=Q, R=R, QN=QN,
            Sigma_w=Sigma_w, x_ini=x_ini, Sigma_ini=Sigma_ini,
            noise_family=noise_family,
            noise_param=float(param),
            N=N,
            n_mc=n_mc,
            seed=seed + i,
        )
        rows.append(
            {
                "param": float(param),
                "mean": float(np.mean(costs)),
                "q95": float(np.quantile(costs, 0.95)),
                "worst": float(np.max(costs)),
            }
        )
    return pd.DataFrame(rows)
