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