import pandas as pd
import casadi as ca
import numpy as np

def process_data(train_set_directory, test_set_directory):
    # Prepare Data
    Xy_train = pd.read_csv(train_set_directory, header=0).values
    Xy_test  = pd.read_csv(test_set_directory, header=0).values

    # X -> other parameters ; y -> quality
    X_train, y_train = Xy_train[:, :-1], Xy_train[:, -1]
    X_test,  y_test  = Xy_test[:, :-1],  Xy_test[:, -1]

    n, p = X_train.shape
    e = np.ones((n, 1)) # Bias Term

    X = ca.DM(np.hstack((e, X_train.astype(float))))
    y = ca.DM(y_train.reshape(-1, 1).astype(float))
    
    return X, y, y_train, X_test, y_test

# Function to check Convexity
def check_convexity(X:ca.DM):
    
    X_np = np.array(X)
    
    num_cols = X_np.shape[1]
    rank = np.linalg.matrix_rank(X_np)
    
    print(f"Matrix X has shape (n, p+1): {X_np.shape}")
    print(f"Number of columns (p+1): {num_cols}")
    print(f"Rank of matrix A: {rank}")
    
    if rank == num_cols:
        print("Result: Rank == Number of columns.")
        print("The matrix A has full column rank.")
        print("Therefore, f(w) is strictly convex for this dataset.")
    else:
        print("Result: Rank < Number of columns.")
        print("The matrix A does not have full column rank.")
        print("Therefore, f(w) is convex, but NOT strictly convex for this dataset.")