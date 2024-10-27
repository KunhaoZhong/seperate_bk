""" KZ todo:
    1. add the extra argument to functions, for example slow-collider template has alpha
"""
import numpy as np
import torch

# kz note: seems VERY hard to learn to the the divergence near 0
def bk_function_local(k1, k2, k3):
    return (k1**2 / (k2 * k3) + k2**2 / (k1 * k3) + k3**2 / (k1 * k2))/3

def bk_function_equilateral(k1, k2, k3):
    return (k1/k2 + k2/k1 + k1/k3 + k3/k1 + k2/k3 + k3/k2)\
         - (k1**2 / (k2 * k3) + k2**2 / (k1 * k3) + k3**2 / (k1 * k2)) - 2

def bk_function_test_1(k1, k2, k3):
    return k1**2 / (k2 * k3)

# 2307.01751: eq 5.15 minus the equilateral part. 
def bk_function_slow_collider_minus_eq(k1, k2, k3, alpha=0.1):
    return   (k1**2/k2/k3) / (1. + (alpha*k1**2/k2/k3)**2) / 3\
            +(k2**2/k1/k3) / (1. + (alpha*k2**2/k1/k3)**2) / 3\
            +(k3**2/k1/k2) / (1. + (alpha*k3**2/k1/k2)**2) / 3

def bk_function_slow_collider_full(k1, k2, k3, alpha=0.1):
    return  bk_function_slow_collider_minus_eq(k1, k2, k3, alpha) + bk_function_equilateral(k1,k2,k3)

def get_function(func_name):
    function_map = {
        'bk_loc': bk_function_local,
        'bk_eq':bk_function_equilateral,
        'bk_test1': bk_function_test_1,
        'bk_sl_collider': bk_function_slow_collider_minus_eq,
        'bk_sl_collider_full': bk_function_slow_collider_full

    }
    return function_map.get(func_name)

def generate_scale_invariant_k_points(n_points_k1=300, kmin = 0.001, kmax=None, n_points_k2 = None):
    """
    Generates k points where:
    1. k3 = 1 (fixed due to scale invariance)
    2. k1 < k2 < k3=1 (ordered)
    3. k3=1 < k1 + k2 (triangle inequality)

    Note: k1 and k2 in this case is k1/k3 and k2/k3 since k3 is normalized to 1 for scale-invariant calculations

    KZ Note: I thought there is a one-liner numpy solution but could not find it

    """
    points = []
    if n_points_k2 is None:
        n_points_k2 = n_points_k1
    if kmax is None:
        kmax = 1.0

    # Since k3=1, and k1 < k2 < 1, we can start small and work up
    for k1 in np.linspace(kmin, kmax, n_points_k1):
        
        # avoid duplicated [1,1,1]
        if k1 ==kmax:
            points.append((k1, k2, 1.0))
            continue
        # k2 must be greater than k1 but less than 1
        for k2 in np.linspace(k1, kmax, n_points_k2):
            # Check triangle inequality: 1 < k1 + k2
            if 1 < k1 + k2:
                points.append((k1, k2, 1.0)) # k3=1.0 in this function, kmax is for k1/k3 and k2/k3
    return np.array(points)


def create_bk_dataset(grid_points, func_name, kmin=0.01, kmax=1.0, n_points_k2=None, scale_invariant=True):
    func = get_function(func_name)
    if scale_invariant:
        ks_scale_invariant = generate_scale_invariant_k_points(n_points_k1=grid_points, kmin=kmin, kmax=kmax, n_points_k2=n_points_k2)
    else:
        raise NotImplementedError
    y = func(ks_scale_invariant[:, 0], ks_scale_invariant[:, 1], ks_scale_invariant[:, 2])
    return torch.tensor(ks_scale_invariant, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)
