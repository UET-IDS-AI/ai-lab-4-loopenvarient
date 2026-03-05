"""
AI_stats_lab.py

Autograded lab: Gradient Descent + Linear Regression (Diabetes)

You must implement the TODO functions below.
Do not change function names or return signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import datasets

# =========================
# Helpers (you may use these)
# =========================

def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Add a bias (intercept) column of ones to X."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return np.hstack([np.ones((X.shape[0], 1)), X])


def standardize_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize using train statistics only.
    Returns: X_train_std, X_test_std, mean, std
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


@dataclass
class GDResult:
    theta: np.ndarray              # (d, )
    losses: np.ndarray             # (T, )
    thetas: np.ndarray             # (T, d) trajectory


# =========================
# Q1: Gradient descent + visualization data
# =========================

def gradient_descent_linreg(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 200,
    theta0: Optional[np.ndarray] = None,
) -> GDResult:
    """
    Linear regression with batch gradient descent on MSE loss.

    X should already include bias column if you want an intercept.

    Returns GDResult with final theta, per-epoch losses, and theta trajectory.
    """
    # TODO: implement
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    y = y.reshape(-1)
    n, d = X.shape
    if y.shape[0] != n:
        raise ValueError("y must have same number of rows as X")

    if theta0 is None:
        theta = np.zeros(d, dtype=float)
    else:
        theta = np.array(theta0, dtype=float).reshape(-1)
        if theta.shape[0] != d:
            raise ValueError("theta0 has wrong shape")

    losses = np.zeros(epochs, dtype=float)
    thetas = np.zeros((epochs, d), dtype=float)

    # MSE: (1/n) ||Xθ - y||^2
    # grad = (2/n) X^T (Xθ - y)
    for t in range(epochs):
        y_pred = X @ theta
        err = y_pred - y
        losses[t] = float(np.mean(err ** 2))
        thetas[t] = theta
        grad = (2.0 / n) * (X.T @ err)
        theta = theta - lr * grad

    return GDResult(theta=theta, losses=losses, thetas=thetas)


def visualize_gradient_descent(
    lr: float = 0.1,
    epochs: int = 60,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Create a small synthetic 2D-parameter problem (bias + 1 feature),
    run gradient descent, and return data needed for visualization.

    Return dict with:
      - "theta_path": (T, 2) array of (theta0, theta1) over time
      - "losses": (T,) loss values
      - "X": design matrix used (with bias) shape (n, 2)
      - "y": targets shape (n,)

    Students can plot:
      - loss curve losses vs epoch
      - theta trajectory in parameter space (theta0 vs theta1)

    Inspired by AML lecture gradient descent trajectory visualization. :contentReference[oaicite:1]{index=1}
    """
    # TODO: implement using gradient_descent_linreg and a synthetic dataset
    rng = np.random.default_rng(seed)
    n = 60
    x = rng.uniform(-1.0, 1.0, size=(n, 1))
    X = add_bias_column(x)

    # True params (bias + slope)
    theta_true = np.array([1.5, -2.0], dtype=float)
    noise = 0.15 * rng.standard_normal(n)
    y = (X @ theta_true) + noise

    res = gradient_descent_linreg(X, y, lr=lr, epochs=epochs, theta0=np.zeros(2))
    return {
        "theta_path": res.thetas.copy(),
        "losses": res.losses.copy(),
        "X": X,
        "y": y.reshape(-1),
    }

def plot_loss_curve(losses):
        df = pd.DataFrame({
        "epoch": range(len(losses)),
        "loss": losses
    })

        df.plot(x="epoch", y="loss")

def plot_theta_trajectory(theta_path):
       df = pd.DataFrame(theta_path, columns=["theta0", "theta1"])

       df.plot(x="theta0", y="theta1")

# =========================
# Q2: Diabetes regression using gradient descent
# =========================

def diabetes_linear_gd(
    lr: float = 0.05,
    epochs: int = 2000,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Load diabetes dataset, split train/test, standardize, fit linear regression via GD.

    Returns:
      train_mse, test_mse, train_r2, test_r2, theta
    """
    # TODO: implement
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    data = load_diabetes()
    X = data.data.astype(float)
    y = data.target.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    X_train_std, X_test_std, _, _ = standardize_train_test(X_train, X_test)
    X_train_b = add_bias_column(X_train_std)
    X_test_b = add_bias_column(X_test_std)

    # Initialize with zeros for determinism
    res = gradient_descent_linreg(
        X_train_b, y_train, lr=lr, epochs=epochs, theta0=np.zeros(X_train_b.shape[1])
    )
    theta = res.theta

    yhat_train = X_train_b @ theta
    yhat_test = X_test_b @ theta

    train_m = mse(y_train, yhat_train)
    test_m = mse(y_test, yhat_test)
    train_r = r2_score(y_train, yhat_train)
    test_r = r2_score(y_test, yhat_test)

    return train_m, test_m, train_r, test_r, theta


# =========================
# Q3: Diabetes regression using analytical solution
# =========================

def diabetes_linear_analytical(
    ridge_lambda: float = 1e-8,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Closed-form solution (normal equation) for linear regression.

    Uses a tiny ridge term (lambda) for numerical stability:
      theta = (X^T X + lambda I)^(-1) X^T y

    Returns:
      train_mse, test_mse, train_r2, test_r2, theta
    """
    # TODO: implement
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    data = load_diabetes()
    X = data.data.astype(float)
    y = data.target.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    X_train_std, X_test_std, _, _ = standardize_train_test(X_train, X_test)
    X_train_b = add_bias_column(X_train_std)
    X_test_b = add_bias_column(X_test_std)

    d = X_train_b.shape[1]
    XtX = X_train_b.T @ X_train_b
    Xty = X_train_b.T @ y_train

    # Ridge for numerical stability; do not penalize bias term
    reg = ridge_lambda * np.eye(d)
    reg[0, 0] = 0.0

    theta = np.linalg.solve(XtX + reg, Xty)

    yhat_train = X_train_b @ theta
    yhat_test = X_test_b @ theta

    train_m = mse(y_train, yhat_train)
    test_m = mse(y_test, yhat_test)
    train_r = r2_score(y_train, yhat_train)
    test_r = r2_score(y_test, yhat_test)

    return train_m, test_m, train_r, test_r, theta



# =========================
# Q4: Compare GD vs analytical
# =========================

def diabetes_compare_gd_vs_analytical(
    lr: float = 0.05,
    epochs: int = 4000,
    test_size: float = 0.2,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Fit diabetes regression using both GD and analytical solution and compare.

    Return dict with:
      - "theta_l2_diff"
      - "train_mse_diff"
      - "test_mse_diff"
      - "train_r2_diff"
      - "test_r2_diff"
      - "theta_cosine_sim"

    (Cosine similarity near 1 means parameters align.)
    """
    # TODO: implement
    gd_train_mse, gd_test_mse, gd_train_r2, gd_test_r2, theta_gd = diabetes_linear_gd(
        lr=lr, epochs=epochs, test_size=test_size, seed=seed
    )

    an_train_mse, an_test_mse, an_train_r2, an_test_r2, theta_an = diabetes_linear_analytical(
        ridge_lambda=1e-8, test_size=test_size, seed=seed
    )

    diff = theta_gd - theta_an
    theta_l2 = float(np.linalg.norm(diff))

    # cosine similarity
    denom = (np.linalg.norm(theta_gd) * np.linalg.norm(theta_an))
    if denom == 0:
        cos_sim = 0.0
    else:
        cos_sim = float(np.dot(theta_gd, theta_an) / denom)

    return {
        "theta_l2_diff": theta_l2,
        "train_mse_diff": float(abs(gd_train_mse - an_train_mse)),
        "test_mse_diff": float(abs(gd_test_mse - an_test_mse)),
        "train_r2_diff": float(abs(gd_train_r2 - an_train_r2)),
        "test_r2_diff": float(abs(gd_test_r2 - an_test_r2)),
        "theta_cosine_sim": cos_sim,
    }
