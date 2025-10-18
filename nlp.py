import casadi as ca
import numpy as np
import  os
from utils import steepest_descent, process_data, check_convexity

# Mean Absolute Error
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

train_set_dir = os.path.join(os.getcwd(),"dataset/Xytrain.csv")
test_set_dir  = os.path.join(os.getcwd(), "dataset/Xytest.csv")

# Load Processed Data
X, y, y_train, X_test, y_test = process_data(train_set_dir, test_set_dir)

# Check Convexity of Dataset
check_convexity(X)

# Simple Model (lambda = 0)
w_star_basic, t_basic, it_basic, fval_basic, gradnorm_basic, rho_basic = steepest_descent(X, y, lam=0)

# L2-Regularized Model (lambda = 5)
w_star_l2, t_l2, it_l2, fval_l2, gradnorm_l2, rho_l2 = steepest_descent(X, y, lam=5)

# Predictions
y_pred_train_basic = np.array(X @ ca.DM(w_star_basic))
y_pred_test_basic  = np.array(np.hstack((np.ones((X_test.shape[0], 1)), X_test)) @ w_star_basic)
y_pred_train_l2 = np.array(X @ ca.DM(w_star_l2))
y_pred_test_l2  = np.array(np.hstack((np.ones((X_test.shape[0], 1)), X_test)) @ w_star_l2)

mae_train_basic = mae(y_train, y_pred_train_basic)
mae_test_basic  = mae(y_test, y_pred_test_basic)
mae_train_l2    = mae(y_train, y_pred_train_l2)
mae_test_l2     = mae(y_test, y_pred_test_l2)

print("\n==== Simple Model (lambda = 0) ====")
print(f"Runtime (s): {t_basic:.4f}")
print(f"Iterations: {it_basic}")
print(f"Final Objective: {fval_basic:.4f}")
print(f"Norm of Gradient: {gradnorm_basic:.4e}")
print(f"Mean Abs Error (Train Set): {mae_train_basic:.4f}")
print(f"Mean Abs Error (Test Set): {mae_test_basic:.4f}")
if rho_basic is not None:
    print(f"Convergence Rate (Rho): {rho_basic:.6f}")

print("\n==== L2-Regularized Model (lambda = 5) ====")
print(f"Runtime (s): {t_l2:.4f}")
print(f"Iterations: {it_l2}")
print(f"Final Objective: {fval_l2:.4f}")
print(f"Norm of Gradient: {gradnorm_l2:.4e}")
print(f"Mean Absolute Error (Train Set): {mae_train_l2:.4f}")
print(f"Mean Absolute Error (Test Set): {mae_test_l2:.4f}")
if rho_l2 is not None:
    print(f"Convergence Rate (Rho): {rho_l2:.6f}")
