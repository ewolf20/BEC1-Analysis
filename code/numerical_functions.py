import mpmath
import numpy as np 


"""
Convenience function which allows lazy evaluation like the ufunc where keyword 
on arbitrary numpy-supporting functions.

Where only two functions are specified in the *funs argument, the function behaves 
exactly as numpy where; true values evaluate the first function, false values the second. 
Where more than two are specified, condition acts as a switch statement; the value condition[i] indicates 
which function should be used to evaluate input[i]. 
"""
def smart_where(condition, input, *funs):
    scalar_condition = np.ndim(condition) == 0
    if scalar_condition:
        condition = np.expand_dims(condition, 0)
    scalar_input = np.ndim(input) == 0
    if scalar_input:
        input = np.expand_dims(input, 0)
    return_array = np.empty_like(input) 
    if(len(funs) < 2):
        raise ValueError("At least two functions must be specified.")
    if(len(funs) == 2):
        funs = (funs[1], funs[0])
    num_funs = int(np.max(condition)) + 1
    for i in range(num_funs):
        return_array[condition == i] = funs[i](input[condition == i])
    if scalar_condition and scalar_input:
        return_array = np.squeeze(return_array)
    return return_array


"""
Convenience function which generates coefficients related to the Taylor expansion of the polylogarithm.

Specifically, the derivative of the polylog at a given point is given by 

d^n/dz^n (Li_s(z)) = 1/z^n \sum_{j = 0}^n a_{nj} Li_{s - j} (z) 

where the a_{nj} have a_{00} = 1, a_{0j} = 0, j > 0, and 

a_{k + 1, j} = -k a_{k, j} + a_{k, j - 1}

This convenience function returns the coefficients that should be multiplied by the polylogarithm values at an arbitrary z, 
plus 1/z^n, to get the _TAYLOR SERIES_ at that z. As such, they include the factorial that divides the coefficients. That is, what is returned 
is 1/(n!) a_{n, j}; this is the case for stability against over/underflow with large numerators and denominators"""

def polylog_taylor_series_coefficient_generator(order):
    number_samples = order + 1 
    return_array = np.zeros((number_samples, number_samples))
    return_array[0][0] = 1 
    for i in range(1, number_samples):
        for j in range(1, number_samples):
            return_array[i][j] = - (i - 1) / i * return_array[i - 1][j] + 1.0 / (i) * return_array[i - 1][j - 1]
    return return_array


vectorized_mpmath_polylog = np.vectorize(mpmath.polylog, otypes = [complex])

"""
Convenience function which generates a series of coefficients related to a taylor centered series at a specific value of the polylogarithm. 

Given center z_0 and order m, returns coefficients b_m given by 

b_m = (1 / m!) \sum_{j = 0}^m a_{mj} Li_{s - j}(z_0)

Note that the 1/z^m dependence is not included; this will be included in the final taylor series against the (z - z_0)^m term so as to combat over/underflow."""
def polylog_specific_taylor_series_generator(center, order, s):
    polylog_taylor_series_coefficients = polylog_taylor_series_coefficient_generator(order) 
    number_terms = order + 1
    s_values = np.linspace(s, s - number_terms, number_terms, endpoint = False)
    polylog_values = vectorized_mpmath_polylog(s_values, center).reshape((1, number_terms)) 
    polylog_specific_taylor_series_coefficients = np.sum(polylog_taylor_series_coefficients * polylog_values, axis = -1)
    return polylog_specific_taylor_series_coefficients


"""
Taylor series which expands the polylog about a center point z0 using pre-generated coefficients as given by polylog_specific_taylor_series_generator."""
def polylog_taylor_series(z, z0, coefficients):
    reshaped_z_over_z0 = np.expand_dims(z / z0, axis = -1)
    number_terms = coefficients.shape[-1]
    term_indices = np.arange(number_terms, dtype = float)
    reshaped_term_indices = np.expand_dims(term_indices, tuple(np.arange(len(z.shape))))
    if(len(coefficients.shape) == 1):
        reshaped_coefficients = np.expand_dims(coefficients, tuple(np.arange(len(z.shape))))
    else:
        reshaped_coefficients = coefficients
    
    terms = np.power((reshaped_z_over_z0 - 1), reshaped_term_indices) * reshaped_coefficients
    return np.real(np.sum(terms, axis = -1))


def generate_and_save_taylor_series_coefficients(s, coeff_save_path, center_save_path):
    START_POINT = -0.1
    END_POINT = 12.0
    POLYLOG_SERIES_ORDER_SMALL = 11
    POLYLOG_SERIES_ORDER_MEDIUM = 11
    POLYLOG_SERIES_ORDER_LARGE = 11
    center_values = -np.logspace(START_POINT, END_POINT, num = 60, base = np.e)
    coefficients_list = []
    for center_value in center_values:
        if(np.log(center_value) < 2):
            coefficients_list.append(polylog_specific_taylor_series_generator(center_value, POLYLOG_SERIES_ORDER_SMALL, s))
        elif(np.log(center_value) < 4):
            coefficients = polylog_specific_taylor_series_generator(center_value, POLYLOG_SERIES_ORDER_MEDIUM, s)
            coefficients_extended = np.append(coefficients, np.zeros(POLYLOG_SERIES_ORDER_SMALL - POLYLOG_SERIES_ORDER_MEDIUM))
            coefficients_list.append(coefficients_extended)
        else:
            coefficients = polylog_specific_taylor_series_generator(center_value, POLYLOG_SERIES_ORDER_LARGE, s)
            coefficients_extended = np.append(coefficients, np.zeros(POLYLOG_SERIES_ORDER_SMALL - POLYLOG_SERIES_ORDER_LARGE))
            coefficients_list.append(coefficients_extended)
    np.save(coeff_save_path, np.stack(coefficients_list))
    np.save(center_save_path, center_values)



def stored_coeffs_polylog_taylor_series(z, center_array, coeff_array):
    reshaped_z = np.expand_dims(z, axis = -1)
    reshaped_center_array = np.expand_dims(center_array, tuple(np.arange(len(z.shape))))
    minimum_indices = np.argmin(np.abs(reshaped_z - reshaped_center_array), axis = -1, keepdims = True)
    center_values_to_use = np.squeeze(np.take_along_axis(reshaped_center_array, minimum_indices, axis = -1))
    coeff_values_to_use = np.squeeze(np.take_along_axis(coeff_array, minimum_indices, axis = 0))
    return polylog_taylor_series(z, center_values_to_use, coeff_values_to_use)



"""General implementation of the cubic formula, useful for fast computation

Given coefficients a, b, c, d of the cubic equation 

ax^3 + bx^2 + cx + d = 0 

where a, b, c, d are assumed to broadcast together, return a cube root of the 
equation of order specified by cube root order. 

Formula is taken from the section "General Cubic Formula" on Wikipedia; the cube root order 
follows the convention of that section. 

Note that the function does not yet handle degeneracies.
"""
def cubic_formula(a, b, c, d, cube_root_order = 0, cast_to_real = False):
    delta_0 = (np.square(b) - 3 * a * c ) * (1 + 0j)
    #Cast delta to complex for safety
    delta_1 = (2 * np.power(b, 3) - 9 * a * b * c + 27 * np.square(a) * d) * (1 + 0j)
    C_fundamental = np.power(1/2.0 * (delta_1 + np.sqrt(np.square(delta_1) - 4 * np.power(delta_0, 3))), 1/3)
    xi = -0.5 + 1j * np.sqrt(3) / 2
    C_specific = C_fundamental * np.power(xi, cube_root_order) 
    root_value = -1/(3 * a) * (b + C_specific + delta_0 / C_specific)
    if cast_to_real:
        return np.real(root_value) 
    else:
        return root_value