import matplotlib.pyplot as plt
import numpy as np


def print_closed_form_report(
    sigma2_lb, b_lb, sigma2_ub, b_ub, eta1, eta2, eta, selected_tau, objective
):
    print("Lower bounds from Lemma 1")
    print(f"  sigma^2 lower bound : {sigma2_lb:.6f}")
    print(f"  b lower bound       : {b_lb:.6f}")
    print()
    print("Upper bounds chosen empirically")
    print(f"  sigma^2 upper bound : {sigma2_ub:.6f}")
    print(f"  b upper bound       : {b_ub:.6f}")
    print()
    print("KL-ball radius from Theorem 1")
    print(f"  eta1 = {eta1:.6f}")
    print(f"  eta2 = {eta2:.6f}")
    print(f"  eta  = {eta:.6f}")
    print()
    print(f"Selected tau = {selected_tau:.6f}")
    print(f"Best outer objective of (17) = {objective:.6f}")


def plot_tau_objective(tau_grid, tau_objective_curve, selected_tau):
    mask = np.isfinite(tau_objective_curve)
    plt.figure()
    plt.plot(tau_grid[mask], tau_objective_curve[mask])
    plt.axvline(selected_tau, linestyle="--", label=f"selected tau = {selected_tau:.3g}")
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\tau (\eta + W_\tau)$")
    plt.title("Fig. 1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_all_metrics_two_by_one(gauss_mm, gauss_lqg, lap_mm, lap_lqg):
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=False)
    metric_style = [
        ("mean", "C0"),
        ("q95", "C1"),
        ("worst", "C2"),
    ]

    panel_spec = [
        (axes[0], gauss_mm, gauss_lqg, "Gaussian Mechanism", r"$\sigma^2$"),
        (axes[1], lap_mm, lap_lqg, "Laplace Mechanism", r"$b$"),
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
        ax.set_ylabel(r"$J(\mathcal{K})$")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))

    plt.show()
