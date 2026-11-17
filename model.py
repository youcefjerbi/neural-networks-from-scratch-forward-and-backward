"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
import numpy as np


def numerical_gradient(f, x, eps=1e-5):
    """Estimate the gradient of scalar-valued f using central differences."""
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    x_work = np.array(x, dtype=float, copy=True)
    gradient = np.zeros_like(x_work)

    for index in np.ndindex(x_work.shape):
        original = x_work[index]

        try:
            x_work[index] = original + eps
            f_plus = f(x_work)

            x_work[index] = original - eps
            f_minus = f(x_work)
        finally:
            x_work[index] = original

        gradient[index] = (f_plus - f_minus) / (2 * eps)

    return gradient

# Step 2 - gradient_check
import numpy as np


def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    analytic = np.asarray(analytic_grad, dtype=float)
    numeric = np.asarray(numeric_grad, dtype=float)

    if analytic.shape != numeric.shape:
        raise ValueError("Gradients must have the same shape")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")

    if analytic.size == 0:
        return 0.0

    difference = np.abs(analytic - numeric)
    denominator = np.maximum(
        tol,
        np.maximum(np.abs(analytic), np.abs(numeric))
    )

    return float(np.max(difference / denominator))

# Step 3 - make_dense
import numpy as np


def make_dense(in_dim, out_dim, weight_init_fn):
    # Call the initializer exactly once.
    W, b = weight_init_fn(in_dim, out_dim)

    if W.shape != (in_dim, out_dim):
        raise ValueError("W must have shape (in_dim, out_dim)")
    if b.shape != (out_dim,):
        raise ValueError("b must have shape (out_dim,)")

    params = {"W": W, "b": b}

    def forward(x):
        y = x @ params["W"] + params["b"]
        cache = x
        return y, cache

    def backward(dout, cache):
        x = cache

        dx = dout @ params["W"].T
        dW = x.T @ dout
        db = np.sum(dout, axis=0)

        grads = {"W": dW, "b": db}
        return dx, grads

    return {
        "params": params,
        "forward": forward,
        "backward": backward,
    }

# Step 4 - make_activation (not yet solved)
# TODO: implement

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

