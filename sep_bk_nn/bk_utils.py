import numpy as np
from scipy import interpolate
from scipy import integrate

def Delta_fNL_scale_w_interp(k_vals, bk_result_1, bk_result_2, kmin, scale_invariant=True, DEBUG=False):
    """
    KZ note: I try to make the integration as defensive as possible for nans. But the interpolater can still give nan. 
            see:https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html#scipy.interpolate.LinearNDInterpolator
            note the irregular of shapes of the interpolation boundaries
            For now I'll just force these points to 0.
            
            In the future maybe consider extrapolation with interpn, but I don't fill safe with that.
            
    Compute integral using interpolation of grid data
    from bk_result_1 and bk_result_2. This would give:
    <bk_result_1-bk_result_2 | bk_result_1 - bk_result_2> / <bk_result_2 | bk_result_2>
    """
    if not scale_invariant:
        raise NotImplementedError

    k1_grid = k_vals[:,0]
    k2_grid = k_vals[:,1]
    k3_grid = k_vals[:,2]
    
    assert np.all(k3_grid == 1), "k3 is not all 1. ?"
    assert bk_result_1.shape[0] == bk_result_2.shape[0], "the two sets of Bk should be evaluated at the same ks"

    # Create interpolation function
    interp_func_1 = interpolate.CloughTocher2DInterpolator(
        list(zip(k1_grid, k2_grid)), bk_result_1
    )

    interp_func_2 = interpolate.CloughTocher2DInterpolator(
        list(zip(k1_grid, k2_grid)), bk_result_2
    )
    
    def integrand_diff(x, y):
        # Square the interpolated function value
        return (interp_func_1(x, y) - interp_func_2(x, y))**2

    def integrand_2(x, y):
        # Square the interpolated function value
        result = (interp_func_2(x, y))**2
        if DEBUG:
            if np.isnan(result):
                print(f"NaN encountered in integrand at x={x:.6f}, y={y:.6f}")
        return result
    
    def integrand_diff_safe(x, y):
        if y >= x and y+x>1. and y <= 1.0 and x > kmin and x < 1.0:  # Only evaluate within the valid region
            result = integrand_diff(x, y)
        else:
            result = 0.0  # Return 0 for points outside our region of interest
        if DEBUG:
            if np.isnan(result):
                print(f"NaN encountered in integrand at x={x:.6f}, y={y:.6f}")
        # KZ note: this is not ideal
        if np.isnan(result):
            result = 0.
        return result
    
    def integrand_2_safe(x, y):
        if y >= x and y+x>1. and y <= 1.0 and x > kmin and x < 1.0:  # Only evaluate within the valid region
            result = integrand_2(x, y)
        else:
            result = 0.0  # Return 0 for points outside our region of interest
        if DEBUG:
            if np.isnan(result):
                print(f"NaN encountered in integrand at x={x:.6f}, y={y:.6f}")
        # KZ note: this is not ideal
        if np.isnan(result):
            result = 0.
        return result
    
    # Inner product of 1 and 2
    IP_1_2, error_1 = integrate.dblquad(
        integrand_diff_safe,
        kmin+1e-6, 
        1.,  # x limits
        lambda x: 0., # note the for stablitiy the limit on y is handeled in integra_XX_safe
        lambda x: 1.
    )

    IP_1_1, error_2 = integrate.dblquad(
        integrand_2_safe,
        kmin+1e-6, 
        1.,  # x limits
        lambda x: 0., # note the for stablitiy the limit on y is handeled in integra_XX_safe
        lambda x: 1.
    )
    
    # KZ TESTING START
    if DEBUG:
        test_x = np.linspace(kmin, 1.0, 100)
        test_y = np.linspace(0, 1.0, 100)
        for x in test_x:
            for y in test_y:
                if y >= 1.0-x and y <= 1.0 and y>x:  # Only test points in your integration region
                    val = integrand_2(x, y)
                    if np.isnan(val):
                        print(f"Found NaN at x={x}, y={y}")
    # KZ TESTING END
    
    Delta_fNL = IP_1_2 / IP_1_1
    print('true <B|B> is', IP_1_1)
    print('The bias estimation of fNL is Delta_fNL = ', Delta_fNL)

    return Delta_fNL