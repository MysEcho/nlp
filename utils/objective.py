import casadi as ca
import numpy as np
from time import perf_counter
from tqdm import tqdm


# Steepest Descent Function
def steepest_descent(X, y, lam=0, max_tolerance=1e-6, max_iter=5000):
    """
    Performs steepest descent with optimal step size.
    """
    
    model = lambda val: "Simple Model" if val == 0  else "L2-Regularized Model"
    
    _ , p = X.shape
    w = np.zeros((p, 1))  # initial guess

    # Define symbolic variable and expressions
    w_sym = ca.MX.sym("w", p, 1)
    res = X @ w_sym - y
    
    # Cost Function (Regularization term dictated by Lambda)
    f_sym = 0.5 * ca.dot(res, res) + lam * ca.dot(w_sym, w_sym)
    
    grad_sym = ca.gradient(f_sym, w_sym)
    hess_sym = ca.hessian(f_sym, w_sym)[0]

    # Create CasADi functions
    f_func = ca.Function("f", [w_sym], [f_sym])
    grad_func = ca.Function("grad", [w_sym], [grad_sym])
    hessian = ca.Function("hessian", [w_sym], [hess_sym])

    grad_prev_norm = None
    start_time = perf_counter()

    print(f"Optimizing for {model(lam)}....")
    # Iteration Loop
    for k in tqdm(range(max_iter)):
        grad_val = np.array(grad_func(w)).astype(float)
        grad_norm = np.linalg.norm(grad_val)
        if grad_norm < max_tolerance:
            print(f"Max Tolerance Reached for {model(lam)}\n")
            break

        H_val = np.array(hessian(w)).astype(float)
        denom = grad_val.T @ (H_val @ grad_val)
        if denom <= 0:
            alpha = 1e-3
        else:
            alpha = float((grad_val.T @ grad_val) / denom)

        # Gradient Descent
        w = w - alpha * grad_val

        if grad_prev_norm is not None:
            rho = grad_norm / grad_prev_norm
        else:
            rho = None
        grad_prev_norm = grad_norm

    runtime = perf_counter() - start_time
    fval = float(f_func(w))
    return w, runtime, k + 1, fval, grad_norm, rho
