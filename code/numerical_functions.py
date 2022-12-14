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
    return_array = np.empty_like(input) 
    if(len(funs) < 2):
        raise ValueError("At least two functions must be specified.")
    if(len(funs) == 2):
        funs = (funs[1], funs[0])
    num_funs = int(max(condition)) + 1
    for i in range(num_funs):
        return_array[condition == i] = funs[i](input[condition == i])
    return return_array