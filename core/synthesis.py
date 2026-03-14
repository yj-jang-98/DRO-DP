import numpy as np


def synthesize_fixed_tau_minimax_vector(
    A, B, C, Q, R, QN, Sigma_w, x_ini, Sigma_ini, sigma2_nom, N, tau
):
    n = A.shape[0]

    Sigma = [Sigma_ini.copy()]
    P = []
    Kest = []

    I = np.eye(n)
    R_inv = np.linalg.inv(R)
    Sigma_v_inv = np.array([[1.0 / sigma2_nom]])

    # Forward Riccati recursion.
    for k in range(N):
        Pk = np.linalg.inv(Sigma[k]) + C.T @ Sigma_v_inv @ C - Q / tau
        try:
            # Infeasible tau values fail one of these definiteness checks and are discarded.
            np.linalg.cholesky(Pk)
        except np.linalg.LinAlgError:
            return None

        P.append(Pk)
        Pk_inv = np.linalg.inv(Pk)
        Kk = A @ Pk_inv @ C.T @ Sigma_v_inv
        Kest.append(Kk)

        Sigma_next = Sigma_w + A @ Pk_inv @ A.T
        try:
            np.linalg.cholesky(Sigma_next)
        except np.linalg.LinAlgError:
            return None
        Sigma.append(Sigma_next)

    # Backward Riccati recursion.
    Pi = [None] * (N + 1)
    Lk = [None] * (N + 1)
    Pi[N] = QN.copy()

    try:
        np.linalg.cholesky(np.linalg.inv(Pi[N]) - Sigma[N] / tau)
    except np.linalg.LinAlgError:
        return None

    for k in range(N - 1, -1, -1):
        Pi_kp1_inv = np.linalg.inv(Pi[k + 1])
        Lkp1 = Pi_kp1_inv + B @ R_inv @ B.T - Sigma_w / tau
        try:
            np.linalg.cholesky(Lkp1)
        except np.linalg.LinAlgError:
            return None

        Lk[k + 1] = Lkp1
        Lkp1_inv = np.linalg.inv(Lkp1)

        Pik = Q + A.T @ Lkp1_inv @ A
        try:
            np.linalg.cholesky(Pik)
        except np.linalg.LinAlgError:
            return None
        Pi[k] = Pik

        try:
            np.linalg.cholesky(np.linalg.inv(Pi[k + 1]) - Sigma_w / tau)
            np.linalg.cholesky(np.linalg.inv(Pi[k]) - Sigma[k] / tau)
        except np.linalg.LinAlgError:
            return None

    # Feedback gains.
    F = []
    for k in range(N):
        # This is the minimax state-feedback gain induced by the coupled Riccati recursions.
        denom = I - Sigma[k] @ Pi[k] / tau
        try:
            denom_inv = np.linalg.inv(denom)
        except np.linalg.LinAlgError:
            return None

        gain = R_inv @ B.T @ np.linalg.inv(Lk[k + 1]) @ A @ denom_inv
        F.append(gain)

    # Compute W_tau.
    try:
        # W_tau is the closed-form outer objective term used in the tau search.
        term1 = (x_ini.T @ np.linalg.inv(np.linalg.inv(Pi[0]) - Sigma_ini / tau) @ x_ini)[0, 0] / (
            2.0 * tau
        )
        _, logdet_Sigma_ini = np.linalg.slogdet(Sigma_ini)
        term2 = -0.5 * logdet_Sigma_ini
        term3 = 0.0
        term4 = 0.0

        for k in range(N):
            S_k = P[k] - C.T @ Sigma_v_inv @ C
            _, logdet_Sigma_kp1 = np.linalg.slogdet(Sigma[k + 1])
            _, logdet_S_k = np.linalg.slogdet(S_k)
            term3 += -0.5 * (logdet_Sigma_kp1 + logdet_S_k)

            Sk_inv = np.linalg.inv(S_k)
            inner_mat = I - (
                Kest[k]
                @ (np.array([[sigma2_nom]]) + C @ Sk_inv @ C.T)
                @ Kest[k].T
                @ np.linalg.inv(np.linalg.inv(Pi[k + 1]) - Sigma[k + 1] / tau)
            ) / tau
            _, logdet_inner = np.linalg.slogdet(inner_mat)
            term4 += -0.5 * logdet_inner

        terminal_term = np.linalg.inv(Sigma[N]) - QN / tau
        _, logdet_term5 = np.linalg.slogdet(terminal_term)
        term5 = -0.5 * logdet_term5

        Wtau = float(term1 + term2 + term3 + term4 + term5)
    except np.linalg.LinAlgError:
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


def synthesize_nominal_lqg_vector(A, B, C, Q, R, QN, Sigma_w, Sigma_ini, sigma2_nom, N):
    Sigma = [Sigma_ini.copy()]
    P = []
    Kest = []
    R_inv = np.linalg.inv(R)
    Sigma_v_inv = np.array([[1.0 / sigma2_nom]])

    for k in range(N):
        Pk = np.linalg.inv(Sigma[k]) + C.T @ Sigma_v_inv @ C
        P.append(Pk)
        Pk_inv = np.linalg.inv(Pk)
        Kk = A @ Pk_inv @ C.T @ Sigma_v_inv
        Kest.append(Kk)
        Sigma.append(Sigma_w + A @ Pk_inv @ A.T)

    Pi = [None] * (N + 1)
    Lk = [None] * (N + 1)
    Pi[N] = QN.copy()

    for k in range(N - 1, -1, -1):
        Lkp1 = np.linalg.inv(Pi[k + 1]) + B @ R_inv @ B.T
        Lk[k + 1] = Lkp1
        Pi[k] = Q + A.T @ np.linalg.inv(Lkp1) @ A

    F = []
    for k in range(N):
        F.append(R_inv @ B.T @ np.linalg.inv(Lk[k + 1]) @ A)

    return {
        "kind": "lqg",
        "Sigma": Sigma,
        "P": P,
        "Kest": Kest,
        "Pi": Pi,
        "Lk": Lk,
        "F": F,
    }


def synthesize_minimax_closed_form_vector(
    A, B, C, Q, R, QN, Sigma_w, x_ini, Sigma_ini, sigma2_nom, N, eta, tau_grid
):
    objective = np.full_like(tau_grid, np.nan, dtype=float)
    best = None

    # Evaluate the closed-form outer objective over a tau grid and keep the best feasible controller.
    for i, tau in enumerate(tau_grid):
        ctrl = synthesize_fixed_tau_minimax_vector(
            A, B, C, Q, R, QN, Sigma_w, x_ini, Sigma_ini, sigma2_nom, N, float(tau)
        )
        if ctrl is None:
            continue
        value = float(tau) * (eta + ctrl["Wtau"])
        objective[i] = value
        if np.isfinite(value) and (best is None or value < best["objective"]):
            best = {"objective": float(value), "controller": ctrl}

    return best, objective
