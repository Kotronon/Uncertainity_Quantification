import chaospy as cp
import matplotlib.pyplot as plt
import numpoly
import numpy as np
import numpy.typing as npt

from utils.helpers import load_reference, simulate


E_ref = -0.43893703
V_ref = 0.00019678

def compute_coefficients(
    nodes: npt.NDArray,
    weights: npt.NDArray,
    polynomials: numpoly.ndpoly,
    target_t: float,
    model_kwargs: dict,
    init_cond: dict,
    mode: str,
) -> npt.NDArray:
    """
    Computes the PCE expansion coefficients using the pseudo-spectral approach.
    """
    # 1. Run the forward model simulation at the quadrature nodes
    t_grid = np.array([0.0, target_t])
    sim_outputs = simulate(t_grid, nodes, model_kwargs, init_cond)
    y_at_T = sim_outputs[:, 1]

    if mode == "manual":
        # Expected value of polynomials squared:
        norms = np.array([np.sum((p(nodes)**2) * weights) for p in polynomials])
        
        coefficients = np.zeros(len(polynomials))
        for i, poly in enumerate(polynomials):
            poly_eval = poly(nodes)
            numerator = np.sum(y_at_T * poly_eval * weights)
            coefficients[i] = numerator / norms[i]

    elif mode == "chaospy":
        # FIX: retall=1 returns a tuple: (fitted_polynomial, fourier_coefficients)
        # We use a throwaway variable '_' for the model and capture the true coefficients array.
        _, coefficients = cp.fit_quadrature(polynomials, nodes, weights, y_at_T, retall=1)
        
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return coefficients


def compute_moments(polynomials: numpoly.ndpoly, coefficients: npt.NDArray, joint_dist: cp.Distribution) -> tuple[float, float]:
    """
    Computes the mean and variance from the PCE coefficients.
    """
    # Create the expansion model by combining polynomials and coefficients
    pce_model = cp.sum(polynomials * coefficients)
    
    # Compute analytical moments from the distribution
    mean = cp.E(pce_model, joint_dist)
    variance = cp.Var(pce_model, joint_dist)
    
    return float(mean), float(variance)


if __name__ == "__main__":
    # Define model parameters
    model_kwargs = {"c": 0.5, "k": 2.0, "f": 0.5}
    init_cond = {"y0": 0.5, "y1": 0.0}
    target_t = 10.0
    
    # Define the parameter uncertainty: \omega ~ U(0.95, 1.05)
    omega_dist = cp.Uniform(0.95, 1.05)
    
    degrees = [1, 2, 3, 4, 5, 6]
    
    results_manual = []
    results_chaospy = []
    
    print(f"{'N':<4} | {'Manual Mean':<12} | {'Manual Var':<12} | {'Chaospy Mean':<12} | {'Chaospy Var':<12}")
    print("-" * 65)

    for N in degrees:
        # Generate orthogonal polynomials up to degree N
        polynomials = cp.generate_expansion(N, omega_dist)
        
        # Generate Gaussian quadrature nodes and weights
        nodes, weights = cp.generate_quadrature(N, omega_dist, rule="gaussian")
        
        # --- Approach 1: Manual ---
        coeffs_manual = compute_coefficients(
            nodes, weights, polynomials, target_t, model_kwargs, init_cond, mode="manual"
        )
        mean_m, var_m = compute_moments(polynomials, coeffs_manual, omega_dist)
        results_manual.append((mean_m, var_m))
        
        # --- Approach 2: Chaospy ---
        coeffs_cp = compute_coefficients(
            nodes, weights, polynomials, target_t, model_kwargs, init_cond, mode="chaospy"
        )
        mean_cp, var_cp = compute_moments(polynomials, coeffs_cp, omega_dist)
        results_chaospy.append((mean_cp, var_cp))
        
        print(f"{N:<4} | {mean_m:<12.6f} | {var_m:<12.6e} | {mean_cp:<12.6f} | {var_cp:<12.6e}")

    # ====================================================================
    # 1. Plotting absolute convergence of Mean and Variance
    # ====================================================================
    means_m = [r[0] for r in results_manual]
    vars_m = [r[1] for r in results_manual]
    
    fig1, ax1_abs = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1_abs[0].plot(degrees, means_m, 'o-', label='PCE Mean', color='blue')
    ax1_abs[0].set_xlabel('Polynomial Degree (N)')
    ax1_abs[0].set_ylabel('Expectation of $y_0(10)$')
    ax1_abs[0].set_title('Expectation Convergence')
    ax1_abs[0].grid(True)
    
    ax1_abs[1].plot(degrees, vars_m, 's-', label='PCE Variance', color='red')
    ax1_abs[1].set_xlabel('Polynomial Degree (N)')
    ax1_abs[1].set_ylabel('Variance of $y_0(10)$')
    ax1_abs[1].set_title('Variance Convergence')
    ax1_abs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

    # ====================================================================
    # 2. Compute Relative Errors against Reference Solution
    # ====================================================================
    E_ref = -0.43893703
    V_ref = 0.00019678

    K_values = np.array(degrees)

    # Extract dynamic expectations over K
    E_app1 = np.array([r[0] for r in results_manual])
    E_app2 = np.array([r[0] for r in results_chaospy])

    # Extract dynamic variances over K
    V_app1 = np.array([r[1] for r in results_manual])
    V_app2 = np.array([r[1] for r in results_chaospy])

    # Formula: |Computed - Reference| / |Reference|
    rel_error_E_app1 = np.abs(E_app1 - E_ref) / np.abs(E_ref)
    rel_error_E_app2 = np.abs(E_app2 - E_ref) / np.abs(E_ref)

    rel_error_V_app1 = np.abs(V_app1 - V_ref) / np.abs(V_ref)
    rel_error_V_app2 = np.abs(V_app2 - V_ref) / np.abs(V_ref)

    # ====================================================================
    # 3. Plot Relative Errors
    # ====================================================================
    fig2, (ax1_rel, ax2_rel) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Expectation Errors
    ax1_rel.semilogy(K_values, rel_error_E_app1, marker='o', label='Approach 1 (Manual)')
    # Using a slightly smaller marker and dashed line for Approach 2 so both are visible if identical
    ax1_rel.semilogy(K_values, rel_error_E_app2, marker='s', linestyle='--', label='Approach 2 (Chaospy)')
    ax1_rel.set_xlabel('Parameter K')
    ax1_rel.set_ylabel('Relative Error')
    ax1_rel.set_title('Relative Error of Expectation $\mathbb{E}[y_0(10)]$')
    ax1_rel.grid(True, which="both", ls="--", alpha=0.7)
    ax1_rel.legend()

    # Plot Variance Errors
    ax2_rel.semilogy(K_values, rel_error_V_app1, marker='o', color='green', label='Approach 1 (Manual)')
    ax2_rel.semilogy(K_values, rel_error_V_app2, marker='s', color='red', linestyle='--', label='Approach 2 (Chaospy)')
    ax2_rel.set_xlabel('Parameter K')
    ax2_rel.set_ylabel('Relative Error')
    ax2_rel.set_title('Relative Error of Variance $\mathbb{V}[y_0(10)]$')
    ax2_rel.grid(True, which="both", ls="--", alpha=0.7)
    ax2_rel.legend()

    plt.tight_layout()
    plt.show()