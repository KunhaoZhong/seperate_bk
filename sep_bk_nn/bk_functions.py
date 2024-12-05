""" KZ todo:
    Done:
    1. add the extra argument to functions, for example slow-collider template has alpha
"""
import numpy as np
import torch
import warnings


# kz note: seems VERY hard to learn due to the the divergence near 0
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

# Dictionary containing function info: function object and default arguments
FUNCTION_MAP = {
    'bk_loc': {
        'func': bk_function_local,
        'default_args': {}
    },
    'bk_eq': {
        'func': bk_function_equilateral,
        'default_args': {}
    },
    'bk_test1': {
        'func': bk_function_test_1,
        'default_args': {}
    },
    'bk_sl_collider': {
        'func': bk_function_slow_collider_minus_eq,
        'default_args': {'alpha': 0.1}
    },
    'bk_sl_collider_full': {
        'func': bk_function_slow_collider_full,
        'default_args': {'alpha': 0.1}
    },
}

def get_function(func_name):
    """
    Returns the function and its default arguments
    """
    if func_name not in FUNCTION_MAP:
        raise ValueError(f"Unknown function: {func_name}")
    return FUNCTION_MAP[func_name]

def generate_scale_invariant_k_points(n_points_k1=300, kmin = 0.001, kmax=None, n_points_k2 = None, k2_sample_version=2):
    """
    Generates k points where:
    1. k3 = 1 (fixed due to scale invariance)
    2. k1 < k2 < k3=1 (ordered)
    3. k3=1 < k1 + k2 (triangle inequality)

    Note: k1 and k2 in this case is k1/k3 and k2/k3 since k3 is normalized to 1 for scale-invariant calculations

    KZ Note: I thought there is a one-liner numpy solution but could not find it

    """
    points = []
        
    # fix the step size according to n_points_k1
    if k2_sample_version==0:
        warnings.warn("This k points sampling would not cover the boundary very well, consider switch to version 2")
        if n_points_k2 is None:
            n_points_k2 = n_points_k1
        if kmax is None:
            kmax = 1.0

        step_size = (kmax-kmin)/n_points_k1
        # Since k3=kmax, and k1 < k2 < kmax, we can start small and work up
        for k1 in np.linspace(kmin, kmax, n_points_k1):
            # avoid duplicated [1,1,1]
            if k1 == kmax:
                points.append((k1, k2, 1.0))
                continue
            # k2 must be greater than k1 but less than 1
            for k2 in np.arange(k1, kmax, step_size):
                if k2==1.:
                    print('kztest')
                # Check triangle inequality: 1 < k1 + k2
                if 1 <= k1 + k2:
                    points.append((k1, k2, 1.0)) # k3=1.0 in this function, kmax is for k1/k3 and k2/k3
        return np.array(points)        
    
    
    # k2 always have the same n-points; this will the grid to be more dense near 1-1-1, which might be helpful in some cases
    elif k2_sample_version==1:
        if n_points_k2 is None:
            n_points_k2 = n_points_k1
        if kmax is None:
            kmax = 1.0

        # Since k3=kmax, and k1 < k2 < kmax, we can start small and work up
        for k1 in np.linspace(kmin, kmax, n_points_k1):
            # avoid duplicated [1,1,1]
            if k1 ==kmax:
                points.append((k1, k2, 1.0))
                continue
            # k2 must be greater than k1 but less than 1
            for k2 in np.linspace(k1, kmax, n_points_k2):
                # Check triangle inequality: 1 < k1 + k2
                if 1 <= k1 + k2:
                    points.append((k1, k2, 1.0)) # k3=1.0 in this function, kmax is for k1/k3 and k2/k3
        return np.array(points)
    
    # sample based on meshgrid
    elif k2_sample_version==2:
        assert kmin>=0 #kmax<=1 # these are relativequantities
        k1,k2 = np.meshgrid(*[np.linspace(kmin, kmax, n_points_k1) for _ in range(2)])

        # kfilt = (np.abs(k2-k1)<=1)&(k2+k1>=1)&(k2>=k1)&(1>=k2)
        kfilt = (np.abs(k2-k1)<=1)&(k2+k1>=1)&(k2>=k1)&(kmax>=k2)

        points = np.asarray([k1[kfilt].ravel(),k2[kfilt].ravel(),np.ones_like(k1[kfilt]).ravel()]).T
        
        return np.array(points)
    
        # default, combining ver0 and ver2 to avoid boundary effects
    elif k2_sample_version=='full_uniform':
        if n_points_k2 is None:
            n_points_k2 = n_points_k1
        if kmax is None:
            kmax = 1.0
            
        n_points_k2 = n_points_k2//2
        n_points_k1 = n_points_k1//2
        ### START ver=0
        step_size = (kmax-kmin)/n_points_k1
        # Since k3=kmax, and k1 < k2 < kmax, we can start small and work up
        for k1 in np.linspace(kmin, kmax, n_points_k1):
            # avoid duplicated [1,1,1]
            if k1 == kmax:
                points.append((k1, k2, 1.0))
                continue
            # k2 must be greater than k1 but less than 1
            for k2 in np.arange(k1, kmax, step_size):
                if k2==1.:
                    print('kztest')
                # Check triangle inequality: 1 < k1 + k2
                if 1 <= k1 + k2:
                    points.append((k1, k2, 1.0)) # k3=1.0 in this function, kmax is for k1/k3 and k2/k3
        points = np.array(points)
        ### END ver=0
        
        ### START ver=1
        assert kmin>=0 #kmax<=1 # these are relativequantities
        k1,k2 = np.meshgrid(*[np.linspace(kmin, kmax, n_points_k1) for _ in range(2)])

        # kfilt = (np.abs(k2-k1)<=1)&(k2+k1>=1)&(k2>=k1)&(1>=k2)
        kfilt = (np.abs(k2-k1)<=1)&(k2+k1>=1)&(k2>=k1)&(kmax>=k2)

        points2 = np.asarray([k1[kfilt].ravel(),k2[kfilt].ravel(),np.ones_like(k1[kfilt]).ravel()]).T
        
        return np.vstack((points, points2))
        ### END ver=1
    else:
        raise NotImplementedError('k points sampleing method not defined')

def create_bk_dataset(grid_points, func_name, func_arg, kmin=0.01, kmax=1.0, n_points_k2=None, scale_invariant=True, k2_sample_version=2):
    func_info = get_function(func_name)
    func = func_info['func']
    func_args_input = func_info['default_args'].copy()
    if func_arg is not None:
            func_args_input.update(func_arg)
    # print('kz testing', func_args_input)
    if scale_invariant:
        ks_scale_invariant = generate_scale_invariant_k_points(n_points_k1=grid_points, kmin=kmin, kmax=kmax, n_points_k2=n_points_k2, k2_sample_version=k2_sample_version)
    else:
        raise NotImplementedError
    y = func(ks_scale_invariant[:, 0], ks_scale_invariant[:, 1], ks_scale_invariant[:, 2], **func_args_input)
    return torch.tensor(ks_scale_invariant, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)
