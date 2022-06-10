import ctypes

import numpy as np 

from _polrot_code import ffi, lib 


def main():
    detuning_1A = 1.0
    detuning_1B = 0.0
    detuning_2A = 1.0 
    detuning_2B = 4.0 
    intensity_A = 0.0
    intensity_B = 0.0 
    linewidth = 100.0
    intensity_sat = 1000.0
    double_buffer_out = ffi.new("double[]", 2)
    od_vector = ffi.new("float[]", [1.0, 2.0])
    result = lib.give_polrot_image(od_vector, detuning_1A, detuning_1B, detuning_2A, 
                                    detuning_2B, linewidth, intensity_A, intensity_B, intensity_sat,
                                    double_buffer_out)
    print(result) 
    print(double_buffer_out[0]) 
    print(double_buffer_out[1])




if __name__ == "__main__":
    main()