import numpy as np
from typing import Callable


def frechet_distance(p: np.ndarray, q: np.ndarray, metric: Callable) -> float:
    """
    Compute the discrete Fréchet distance between two (open) curves.

    Parameters
    ----------
    p : np.ndarray
        First curve as an array of shape (n_points_p, dim).
    q : np.ndarray
        Second curve as an array of shape (n_points_q, dim).
    metric : callable
        Function that computes pairwise distances between points of p and q.
        Should accept two arrays and return a (n_points_p, n_points_q) array.

    Returns
    -------
    float
        The discrete Fréchet distance between the two curves.
    """
    P = p.shape[0]
    Q = q.shape[0]

    v = metric(p[0], q)
    v = np.maximum.accumulate(v)

    for i in range(1, P):
        u = np.minimum(v[:-1], v[1:])

        v[0] = max(v[0], metric(p[i], q[0]))
        v[1:] = metric(p[i], q[1:])
        for j in range(1, Q):
            v[j] = max(min(v[j - 1], u[j - 1]), v[j])

    return v[-1]


def frechet_distance_closed_curves(p: np.ndarray, q: np.ndarray, metric: Callable) -> float:
    """
    Compute the discrete Fréchet distance between two closed curves.

    Parameters
    ----------
    p : np.ndarray
        First curve as an array of shape (n_points_p, dim).
    q : np.ndarray
        Second curve as an array of shape (n_points_q, dim).
    metric : callable
        Function that computes pairwise distances between points of p and q.
        Should accept two arrays and return a (n_points_p, n_points_q) array.

    Returns
    -------
    float
        The discrete Fréchet distance between the two closed curves (loops).
    """

    min_dist = float("inf")

    # Calculate the Fréchet distance for every cyclic shift of q
    for i in range(len(q)):
        q_shifted = np.roll(q, shift=-i, axis=0)
        dist = frechet_distance(p, q_shifted, metric)
        if dist < min_dist:
            min_dist = dist

    return min_dist


# For less than 100 vertices the numpy version is faster
def frechet_distance_closed_curves_torch(p, q, metric):
    raise NotImplementedError("Torch version not implemented yet.")


if __name__ == "__main__":
    p = np.array([[1, 2], [3, 4]])
    q = np.array([[2, 1], [3, 3], [5, 5]])
    # Uses the Euclidean distance as the metric
    fd = frechet_distance(p, q, metric=lambda a, b: np.linalg.norm(a - b, axis=-1))
    print(f"Fréchet distance between open curves p and q: {fd}")
    p = np.array([[0, 0], [1, 2], [3, 4]])
    fd = frechet_distance_closed_curves(p, q, metric=lambda a, b: np.linalg.norm(a - b, axis=-1))
    print(f"Fréchet distance between closed curves p and q: {fd}")
