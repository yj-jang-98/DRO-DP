# Auto-generated from minimax_private_lqg_toy.ipynb


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

plt.rcParams["figure.figsize"] = (7.0, 4.5)
np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.precision", 4)



# ---------------------------
# User-tunable configuration
# ---------------------------

SYSTEM = {
    "A": np.array([[1.1, 0.1],
                   [0.0, 0.9]]),    # 2x2
    "B": np.array([[0.1],
                   [1.0]]),         # 2x1
    "C": np.array([[1.0, 0.0]]),     # 1x2
    "W": np.eye(2) * 0.01,           # process noise cov (2x2)
    "V": np.array([[0.01]]),         # measurement noise cov (1x1)
}

# initial state / cov
x0 = np.zeros((2,))                  # 2-dimensional state
P0 = np.eye(2) * 0.1

PRIVACY = {
    "epsilon": 8.0,   # tuned: keep Gaussian q95 close while minimax improves Laplace q95/worst
    "delta": 1e-2,
    "gamma": 1.0,
    "rho_sigma": 1.05, # must be > 1: sigma_bar^2 = rho_sigma * sigma_lb^2
    "rho_b": 1.2,     # b_bar = rho_b * b_lb
}

SIM = {
    "N": 20,
    "n_mc": 2000,     # Monte Carlo runs per point
    "n_grid": 12,     # number of noise-parameter points in each family
    "seed_gaussian": 123,
    "seed_laplace": 456,
}

TAU_MODE = "closed_form"          # "closed_form" or "empirical_worst_case"
TAU_GRID_CLOSED_FORM = np.logspace(-2, 4, 800)
TAU_GRID_EMPIRICAL = np.logspace(-2, 4, 120)
N_MC_TAU_EMPIRICAL = 400



def privacy_lower_bounds_scalar(C, epsilon, delta, gamma):
    sigma2_lb = 2.0 * np.log(1.25 / delta) * (abs(C) ** 2) * (gamma ** 2) / (epsilon ** 2)
    b_lb = abs(C) * gamma / epsilon
    return float(sigma2_lb), float(b_lb)


def kl_radius_scalar(sigma2_nom, sigma2_ub, b_lb, b_ub, L):
    def gfun(x):
        return np.log(sigma2_nom / x) + x / sigma2_nom

    eta1 = float(gfun(sigma2_ub) - 1.0)
    eta2 = float(max(gfun(2.0 * b_lb ** 2), gfun(2.0 * b_ub ** 2)) - 2.0 + np.log(np.pi))
    eta = 0.5 * L * max(eta1, eta2)
    return float(eta), eta1, eta2


def synthesize_fixed_tau_minimax_scalar(A, B, C, Q, R, QN, Sigma_w, x_ini, Sigma_ini, sigma2_nom, N, tau):
    """
    Scalar specialization of the fixed-tau risk-sensitive/minimax controller.

    Notes:
    - Forward Riccati and estimator recursions come from Proposition 2.
    - The outer objective uses the scalar specialization of Eq. (19).
    - Returns None if the current tau is not feasible.
    """
    Sigma = [float(Sigma_ini)]
    P = []
    Kest = []

    # Forward Riccati recursion
    for k in range(N):
        Pk = 1.0 / Sigma[k] + (C ** 2) / sigma2_nom - Q / tau
        if Pk <= 1e-12 or Sigma[k] <= 1e-12:
            return None

        P.append(float(Pk))
        Kk = A * (1.0 / Pk) * C / sigma2_nom
        Kest.append(float(Kk))

        Sigma_next = Sigma_w + A ** 2 * (1.0 / Pk)
        if Sigma_next <= 1e-12:
            return None
        Sigma.append(float(Sigma_next))

    # Backward Riccati recursion
    Pi = [None] * (N + 1)
    Lk = [None] * (N + 1)
    Pi[N] = float(QN)

    if (1.0 / Pi[N] - Sigma[N] / tau) <= 1e-12:
        return None

    for k in range(N - 1, -1, -1):
        Lkp1 = 1.0 / Pi[k + 1] + (B ** 2) / R - Sigma_w / tau
        if Lkp1 <= 1e-12:
            return None
        Lk[k + 1] = float(Lkp1)

        Pik = Q + A ** 2 * (1.0 / Lkp1)
        if Pik <= 1e-12:
            return None
        Pi[k] = float(Pik)

        if (1.0 / Pi[k + 1] - Sigma_w / tau) <= 1e-12:
            return None
        if (1.0 / Pi[k] - Sigma[k] / tau) <= 1e-12:
            return None

    # Feedback gains
    F = []
    for k in range(N):
        denom = 1.0 - Sigma[k] * Pi[k] / tau
        if denom <= 1e-12:
            return None
        gain = (1.0 / R) * B * (1.0 / Lk[k + 1]) * A / denom
        F.append(float(gain))

    # Scalar specialization of Eq. (19) for W_tau
    try:
        term1 = (x_ini ** 2) * (1.0 / (1.0 / Pi[0] - Sigma_ini / tau)) / (2.0 * tau)
        term2 = -0.5 * np.log(Sigma_ini)
        term3 = 0.0
        term4 = 0.0

        for k in range(N):
            S_k = P[k] - (C ** 2) / sigma2_nom
            if S_k <= 1e-12 or Sigma[k + 1] <= 1e-12:
                return None

            term3 += -0.5 * np.log(Sigma[k + 1] * S_k)

            inner = 1.0 - (
                Kest[k]
                * (sigma2_nom + (C ** 2) / S_k)
                * Kest[k]
                * (1.0 / (1.0 / Pi[k + 1] - Sigma[k + 1] / tau))
            ) / tau

            if inner <= 1e-12:
                return None
            term4 += -0.5 * np.log(inner)

        terminal_term = 1.0 / Sigma[N] - QN / tau
        if terminal_term <= 1e-12:
            return None
        term5 = -0.5 * np.log(terminal_term)

        Wtau = float(term1 + term2 + term3 + term4 + term5)

    except (FloatingPointError, ValueError, ZeroDivisionError):
        return None

    return {
        "kind": "minimax",
        "tau": float(tau),
        "Sigma": Sigma,
        "P": P,
        "Kest": Kest,
        "Pi": Pi,
        "Lk": Lk,
        "F": F,
        "Wtau": Wtau,
    }


def synthesize_nominal_lqg_scalar(A, B, C, Q, R, QN, Sigma_w, Sigma_ini, sigma2_nom, N):
    """
    Standard finite-horizon LQG with the nominal Gaussian privacy-noise variance sigma2_nom.
    """
    Sigma = [float(Sigma_ini)]
    P = []
    Kest = []

    for k in range(N):
        Pk = 1.0 / Sigma[k] + (C ** 2) / sigma2_nom
        if Pk <= 1e-12:
            return None

        P.append(float(Pk))
        Kk = A * (1.0 / Pk) * C / sigma2_nom
        Kest.append(float(Kk))
        Sigma.append(float(Sigma_w + A ** 2 * (1.0 / Pk)))

    Pi = [None] * (N + 1)
    Lk = [None] * (N + 1)
    Pi[N] = float(QN)

    for k in range(N - 1, -1, -1):
        Lkp1 = 1.0 / Pi[k + 1] + (B ** 2) / R
        Lk[k + 1] = float(Lkp1)
        Pi[k] = float(Q + A ** 2 * (1.0 / Lkp1))

    F = []
    for k in range(N):
        F.append(float((1.0 / R) * B * (1.0 / Lk[k + 1]) * A))

    return {
        "kind": "lqg",
        "Sigma": Sigma,
        "P": P,
        "Kest": Kest,
        "Pi": Pi,
        "Lk": Lk,
        "F": F,
    }


def synthesize_minimax_closed_form_scalar(A, B, C, Q, R, QN, Sigma_w, x_ini, Sigma_ini, sigma2_nom, N, eta, tau_grid):
    objective = np.full_like(tau_grid, np.nan, dtype=float)
    best = None

    for i, tau in enumerate(tau_grid):
        ctrl = synthesize_fixed_tau_minimax_scalar(
            A=A, B=B, C=C, Q=Q, R=R, QN=QN, Sigma_w=Sigma_w,
            x_ini=x_ini, Sigma_ini=Sigma_ini, sigma2_nom=sigma2_nom,
            N=N, tau=float(tau)
        )
        if ctrl is None:
            continue

        value = float(tau) * (eta + ctrl["Wtau"])
        objective[i] = value

        if np.isfinite(value) and (best is None or value < best["objective"]):
            best = {"objective": float(value), "controller": ctrl}

    return best, objective


def simulate_controller_scalar_vectorized(controller, A, B, C, Q, R, QN, Sigma_w, x_ini, Sigma_ini,
                                          noise_family, noise_param, N, n_mc=1000, seed=0):
    """
    Vectorized Monte Carlo simulation for the scalar system.
    """
    rng = np.random.default_rng(seed)

    x = rng.normal(x_ini, math.sqrt(Sigma_ini), size=n_mc)
    xhat = np.full(n_mc, float(x_ini))
    cost = np.zeros(n_mc)

    for k in range(N):
        u = -controller["F"][k] * xhat
        cost += 0.5 * (Q * x ** 2 + R * u ** 2)

        if noise_family == "gaussian":
            v = rng.normal(0.0, math.sqrt(noise_param), size=n_mc)
        elif noise_family == "laplace":
            v = rng.laplace(0.0, noise_param, size=n_mc)
        else:
            raise ValueError("noise_family must be 'gaussian' or 'laplace'")

        ytilde = C * x + v

        if controller["kind"] == "minimax":
            tau = controller["tau"]
            xhat = (
                A * xhat
                + B * u
                + controller["Kest"][k] * (ytilde - C * xhat)
                + A * (1.0 / controller["P"][k]) * (Q / tau) * xhat
            )
        else:
            xhat = A * xhat + B * u + controller["Kest"][k] * (ytilde - C * xhat)

        w = rng.normal(0.0, math.sqrt(Sigma_w), size=n_mc)
        x = A * x + B * u + w

    cost += 0.5 * QN * x ** 2
    return cost


def tune_tau_empirical_worst_case_scalar(A, B, C, Q, R, QN, Sigma_w, x_ini, Sigma_ini, sigma2_nom, N,
                                         noise_grid, tau_grid, n_mc_inner=400, seed=123):
    """
    Optional fallback:
    choose tau by minimizing the empirical worst-case mean cost over the extreme points
    of the ambiguity set. This is NOT the paper's theorem, but it is useful as a practical
    diagnostic for tiny toy examples.
    """
    objective = np.full_like(tau_grid, np.nan, dtype=float)
    best = None

    for i, tau in enumerate(tau_grid):
        ctrl = synthesize_fixed_tau_minimax_scalar(
            A=A, B=B, C=C, Q=Q, R=R, QN=QN, Sigma_w=Sigma_w,
            x_ini=x_ini, Sigma_ini=Sigma_ini, sigma2_nom=sigma2_nom,
            N=N, tau=float(tau)
        )
        if ctrl is None:
            continue

        wc = -np.inf
        for fam, param in noise_grid:
            costs = simulate_controller_scalar_vectorized(
                controller=ctrl,
                A=A, B=B, C=C, Q=Q, R=R, QN=QN, Sigma_w=Sigma_w,
                x_ini=x_ini, Sigma_ini=Sigma_ini,
                noise_family=fam, noise_param=float(param),
                N=N, n_mc=n_mc_inner, seed=seed
            )
            wc = max(wc, float(np.mean(costs)))

        objective[i] = wc

        if best is None or wc < best["objective"]:
            best = {"objective": float(wc), "controller": ctrl}

    return best, objective


def evaluate_grid(controller, A, B, C, Q, R, QN, Sigma_w, x_ini, Sigma_ini,
                  noise_family, param_grid, N, n_mc=1000, seed=0):
    rows = []

    for i, param in enumerate(param_grid):
        costs = simulate_controller_scalar_vectorized(
            controller=controller,
            A=A, B=B, C=C, Q=Q, R=R, QN=QN, Sigma_w=Sigma_w,
            x_ini=x_ini, Sigma_ini=Sigma_ini,
            noise_family=noise_family, noise_param=float(param),
            N=N, n_mc=n_mc, seed=seed + i
        )

        rows.append({
            "param": float(param),
            "mean": float(np.mean(costs)),
            "q95": float(np.quantile(costs, 0.95)),
            "worst": float(np.max(costs)),
        })

    return pd.DataFrame(rows)


def summarize_family(df_mm, df_lqg, family_name):
    rows = []
    for metric in ["mean", "q95", "worst"]:
        rows.append({
            "family": family_name,
            "metric": metric,
            "worst_over_grid_minimax": float(df_mm[metric].max()),
            "worst_over_grid_lqg": float(df_lqg[metric].max()),
            "best_over_grid_minimax": float(df_mm[metric].min()),
            "best_over_grid_lqg": float(df_lqg[metric].min()),
        })
    return pd.DataFrame(rows)


def plot_metric(df_mm, df_lqg, metric, xlabel, title_prefix):
    plt.figure()
    plt.plot(df_mm["param"], df_mm[metric], marker="o", label="Minimax private LQG")
    plt.plot(df_lqg["param"], df_lqg[metric], marker="o", label="Nominal Gaussian LQG")
    plt.xlabel(xlabel)
    plt.ylabel(metric)
    plt.title(f"{title_prefix}: {metric}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_all_metrics_two_by_one(gauss_mm, gauss_lqg, lap_mm, lap_lqg):
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    metric_style = [
        ("mean", "C0"),
        ("q95", "C1"),
        ("worst", "C2"),
    ]

    panel_spec = [
        (axes[0], gauss_mm, gauss_lqg, "Gaussian Truth", "true Gaussian variance"),
        (axes[1], lap_mm, lap_lqg, "Laplace Truth", "true Laplace scale"),
    ]

    for ax, df_mm, df_lqg, title, xlabel in panel_spec:
        for metric, color in metric_style:
            ax.plot(
                df_mm["param"],
                df_mm[metric],
                marker="o",
                linestyle="-",
                color=color,
                label=f"Minimax {metric}",
            )
            ax.plot(
                df_lqg["param"],
                df_lqg[metric],
                marker="x",
                linestyle="--",
                color=color,
                label=f"LQG {metric}",
            )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mean ~~ q95 ~~ worst")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        ax.legend(ncol=2)

    plt.show()



sigma2_lb, b_lb = privacy_lower_bounds_scalar(
    C=SYSTEM["C"],
    epsilon=PRIVACY["epsilon"],
    delta=PRIVACY["delta"],
    gamma=PRIVACY["gamma"],
)

sigma2_ub = PRIVACY["rho_sigma"] * sigma2_lb
b_ub = PRIVACY["rho_b"] * b_lb

eta, eta1, eta2 = kl_radius_scalar(
    sigma2_nom=sigma2_lb,
    sigma2_ub=sigma2_ub,
    b_lb=b_lb,
    b_ub=b_ub,
    L=SIM["N"] + 1,  # scalar output => p = 1, so L = N + 1
)

print("Privacy bounds from Proposition 1")
print(f"  sigma^2 lower bound : {sigma2_lb:.6f}")
print(f"  b lower bound       : {b_lb:.6f}")
print()
print("Upper bounds chosen for the toy ambiguity set")
print(f"  sigma^2 upper bound : {sigma2_ub:.6f}")
print(f"  b upper bound       : {b_ub:.6f}")
print()
print("KL-ball quantities from Theorem 1")
print(f"  eta1 = {eta1:.6f}")
print(f"  eta2 = {eta2:.6f}")
print(f"  eta  = {eta:.6f}")

nominal_lqg = synthesize_nominal_lqg_scalar(
    A=SYSTEM["A"], B=SYSTEM["B"], C=SYSTEM["C"],
    Q=SYSTEM["Q"], R=SYSTEM["R"], QN=SYSTEM["QN"],
    Sigma_w=SYSTEM["Sigma_w"], Sigma_ini=SYSTEM["Sigma_ini"],
    sigma2_nom=sigma2_lb, N=SIM["N"],
)

if TAU_MODE == "closed_form":
    tau_solution, tau_objective_curve = synthesize_minimax_closed_form_scalar(
        A=SYSTEM["A"], B=SYSTEM["B"], C=SYSTEM["C"],
        Q=SYSTEM["Q"], R=SYSTEM["R"], QN=SYSTEM["QN"],
        Sigma_w=SYSTEM["Sigma_w"], x_ini=SYSTEM["x_ini"], Sigma_ini=SYSTEM["Sigma_ini"],
        sigma2_nom=sigma2_lb, N=SIM["N"], eta=eta,
        tau_grid=TAU_GRID_CLOSED_FORM,
    )
    tau_xlabel = "tau"
    tau_ylabel = "tau * (eta + W_tau)"
else:
    extreme_noise_grid = [
        ("gaussian", sigma2_lb),
        ("gaussian", sigma2_ub),
        ("laplace", b_lb),
        ("laplace", b_ub),
    ]
    tau_solution, tau_objective_curve = tune_tau_empirical_worst_case_scalar(
        A=SYSTEM["A"], B=SYSTEM["B"], C=SYSTEM["C"],
        Q=SYSTEM["Q"], R=SYSTEM["R"], QN=SYSTEM["QN"],
        Sigma_w=SYSTEM["Sigma_w"], x_ini=SYSTEM["x_ini"], Sigma_ini=SYSTEM["Sigma_ini"],
        sigma2_nom=sigma2_lb, N=SIM["N"], noise_grid=extreme_noise_grid,
        tau_grid=TAU_GRID_EMPIRICAL, n_mc_inner=N_MC_TAU_EMPIRICAL, seed=SIM["seed_gaussian"],
    )
    tau_xlabel = "tau"
    tau_ylabel = "empirical worst-case mean cost"

if tau_solution is None:
    raise RuntimeError("No feasible tau was found on the grid. Try enlarging the tau grid or relaxing the toy setup.")

minimax_ctrl = tau_solution["controller"]

print()
print(f"Selected tau ({TAU_MODE}) = {minimax_ctrl['tau']:.6f}")
if TAU_MODE == "closed_form":
    print(f"Best outer objective      = {tau_solution['objective']:.6f}")
else:
    print(f"Best empirical worst-case = {tau_solution['objective']:.6f}")

plt.figure()
if TAU_MODE == "closed_form":
    tau_grid_used = TAU_GRID_CLOSED_FORM
else:
    tau_grid_used = TAU_GRID_EMPIRICAL

mask = np.isfinite(tau_objective_curve)
plt.semilogx(tau_grid_used[mask], tau_objective_curve[mask], marker=".")
plt.axvline(minimax_ctrl["tau"], linestyle="--", label=f"selected tau = {minimax_ctrl['tau']:.3g}")
plt.xlabel(tau_xlabel)
plt.ylabel(tau_ylabel)
plt.title("Outer tau objective")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

gaussian_grid = np.linspace(sigma2_lb, sigma2_ub, SIM["n_grid"])
laplace_grid = np.linspace(b_lb, b_ub, SIM["n_grid"])

gaussian_minimax = evaluate_grid(
    controller=minimax_ctrl,
    A=SYSTEM["A"], B=SYSTEM["B"], C=SYSTEM["C"],
    Q=SYSTEM["Q"], R=SYSTEM["R"], QN=SYSTEM["QN"],
    Sigma_w=SYSTEM["Sigma_w"], x_ini=SYSTEM["x_ini"], Sigma_ini=SYSTEM["Sigma_ini"],
    noise_family="gaussian", param_grid=gaussian_grid,
    N=SIM["N"], n_mc=SIM["n_mc"], seed=SIM["seed_gaussian"],
)

gaussian_lqg = evaluate_grid(
    controller=nominal_lqg,
    A=SYSTEM["A"], B=SYSTEM["B"], C=SYSTEM["C"],
    Q=SYSTEM["Q"], R=SYSTEM["R"], QN=SYSTEM["QN"],
    Sigma_w=SYSTEM["Sigma_w"], x_ini=SYSTEM["x_ini"], Sigma_ini=SYSTEM["Sigma_ini"],
    noise_family="gaussian", param_grid=gaussian_grid,
    N=SIM["N"], n_mc=SIM["n_mc"], seed=SIM["seed_gaussian"],
)

laplace_minimax = evaluate_grid(
    controller=minimax_ctrl,
    A=SYSTEM["A"], B=SYSTEM["B"], C=SYSTEM["C"],
    Q=SYSTEM["Q"], R=SYSTEM["R"], QN=SYSTEM["QN"],
    Sigma_w=SYSTEM["Sigma_w"], x_ini=SYSTEM["x_ini"], Sigma_ini=SYSTEM["Sigma_ini"],
    noise_family="laplace", param_grid=laplace_grid,
    N=SIM["N"], n_mc=SIM["n_mc"], seed=SIM["seed_laplace"],
)

laplace_lqg = evaluate_grid(
    controller=nominal_lqg,
    A=SYSTEM["A"], B=SYSTEM["B"], C=SYSTEM["C"],
    Q=SYSTEM["Q"], R=SYSTEM["R"], QN=SYSTEM["QN"],
    Sigma_w=SYSTEM["Sigma_w"], x_ini=SYSTEM["x_ini"], Sigma_ini=SYSTEM["Sigma_ini"],
    noise_family="laplace", param_grid=laplace_grid,
    N=SIM["N"], n_mc=SIM["n_mc"], seed=SIM["seed_laplace"],
)

summary = pd.concat(
    [
        summarize_family(gaussian_minimax, gaussian_lqg, "Gaussian truth"),
        summarize_family(laplace_minimax, laplace_lqg, "Laplace truth"),
    ],
    ignore_index=True,
)

display(summary)

plot_all_metrics_two_by_one(
    gauss_mm=gaussian_minimax,
    gauss_lqg=gaussian_lqg,
    lap_mm=laplace_minimax,
    lap_lqg=laplace_lqg,
)

print("Gaussian truth - minimax")
display(gaussian_minimax)

print("Gaussian truth - nominal LQG")
display(gaussian_lqg)

print("Laplace truth - minimax")
display(laplace_minimax)

print("Laplace truth - nominal LQG")
display(laplace_lqg)
