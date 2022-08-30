from cffi import FFI 



source_string = r"""
    #include <math.h>

    static double imaging_od_lorentzian(double detuning, double linewidth, double intensity, double intensity_sat) {
        return 1.0 / (1.0 + pow(2.0 * detuning / linewidth, 2.0) + pow(intensity / intensity_sat, 2.0));
    }

    static int give_polrot_image(float *od_vector, double detuning_1A, double detuning_1B, double detuning_2A, double detuning_2B, 
                                double linewidth, double intensity_A, double intensity_B, double intensity_sat, double phase_sign,
                                double *buff_out) {
        double od_naught_1 = od_vector[0];
        double od_naught_2 = od_vector[1];
        double od_1A = od_naught_1 * imaging_od_lorentzian(detuning_1A, linewidth, intensity_A, intensity_sat);
        double od_1B = od_naught_1 * imaging_od_lorentzian(detuning_1B, linewidth, intensity_A, intensity_sat);
        double od_2A = od_naught_2 * imaging_od_lorentzian(detuning_2A, linewidth, intensity_B, intensity_sat);
        double od_2B = od_naught_2 * imaging_od_lorentzian(detuning_2B, linewidth, intensity_B, intensity_sat);
        double phi_A = (- detuning_1A / linewidth * od_1A - detuning_2A / linewidth * od_2A) * phase_sign;
        double phi_B = (- detuning_1B / linewidth * od_1B - detuning_2B / linewidth * od_2B) * phase_sign;
        double abs_A = exp(-od_1A / 2.0) * exp(-od_2A / 2.0);
        double abs_B = exp(-od_1B / 2.0) * exp(-od_2B / 2.0);
        double result_A = 0.5 + pow(abs_A, 2.0) / 2 - abs_A * sin(phi_A);
        double result_B = 0.5 + pow(abs_B, 2.0) / 2 - abs_B * sin(phi_B);
        buff_out[0] = result_A;
        buff_out[1] = result_B;
        return 0;
    }
"""

ffibuilder = FFI()
ffibuilder.cdef("""int give_polrot_image(float *, double detuning_1A, double detuning_1B, double detuning_2A, double detuning_2B, 
                                        double linewidth, double intensity_A, double intensity_B, double intensity_sat, double phase_sign,
                                         double *);""")

ffibuilder.set_source("_polrot_code", source_string)


if __name__ == "__main__":
    ffibuilder.compile(verbose = True) 


"""

        """