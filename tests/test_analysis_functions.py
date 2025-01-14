from collections import namedtuple
import json
import os 
import sys 
import shutil


import astropy
import numpy as np
import scipy

#Temp import 
import matplotlib.pyplot as plt


path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)
from BEC1_Analysis.code import measurement, image_processing_functions, science_functions, data_fitting_functions, eos_functions
from BEC1_Analysis.code import analysis_functions


TEMP_WORKSPACE_PATH = "./resources"
TEMP_MEASUREMENT_FOLDER_NAME = "analysis_test_temp_measurement"

EPOCH_TIMESTRING_PARAMETERS = "1970-01-01T00:00:00"
EPOCH_TIMESTRING_FILENAME = "1970-01-01--00-00-00"

BASE_LIGHT_LEVEL = 5000
BASE_DARK_LEVEL = 100

DEFAULT_ABSORPTION = 1.0/np.e
DEFAULT_ABS_IMAGE_SHAPE = (512, 512)     
DEFAULT_ABS_SQUARE_WIDTH = 127
DEFAULT_ABS_SQUARE_CENTER_INDICES = (256, 256)

FOURIER_SAMPLE_ORDER = 1
FOURIER_SAMPLE_AMPLITUDE = 0.1

RR_CONDENSATE_MAGNITUDE = 0.5
RR_CONDENSATE_WIDTH = 16
RR_THERMAL_MAGNITUDE = 0.5
RR_THERMAL_WIDTH = 64


DEFAULT_ABSORPTION_IMAGE_ROI = [171, 171, 342, 342]
DEFAULT_ABSORPTION_IMAGE_ROI_SHAPE = (171, 171)
EXPANDED_ABSORPTION_IMAGE_ROI = [50, 50, 462, 462]
DEFAULT_ABSORPTION_IMAGE_CLOSE_ROI = [193, 193, 320, 320]
DEFAULT_ABSORPTION_IMAGE_NORM_BOX = [50, 50, 140, 140]
DEFAULT_ABSORPTION_IMAGE_NORM_BOX_SHAPE = (90, 90)
EXPANDED_ABSORPTION_IMAGE_NORM_BOX = [30, 30, 35, 35]
HYBRID_AVOID_ZERO_ROI = [100, 200, 420, 320]
DENSITY_COM_OFFSET = 5




#Dummy values to inject into code instead of the multiplicative and additive identities, to make sure 
#that subtractions have the right signs, divisions are done appropriately, etc. 
L33T_DUMMY = 1.337
PI_DUMMY = 3.14
E_DUMMY = 2.718
SQRT_2_DUMMY = 1.414
SQRT_3_DUMMY = 1.732
SQRT_5_DUMMY = 2.236
SQRT_6_DUMMY = 2.449
SQRT_7_DUMMY = 2.646
SQRT_10_DUMMY = 3.162
SQRT_11_DUMMY = 3.317
SQRT_13_DUMMY = 3.606
SQRT_14_DUMMY = 3.742

class DummyMeasurement(object):
    pass

def _get_raw_pixels_test_helper(type_name, function_to_test):
    try:
        measurement_pathname, my_measurement, my_run = create_measurement(type_name, ROI = DEFAULT_ABSORPTION_IMAGE_ROI)
        returned_array_no_roi = function_to_test(my_measurement, my_run, crop_roi = False) 
        returned_array_roi = function_to_test(my_measurement, my_run, crop_roi = True) 
        with_atoms_no_roi, without_atoms_no_roi, dark_no_roi = returned_array_no_roi 
        assert with_atoms_no_roi.dtype == np.ushort
        assert without_atoms_no_roi.dtype == np.ushort 
        assert dark_no_roi.dtype == np.ushort
        assert with_atoms_no_roi.shape == DEFAULT_ABS_IMAGE_SHAPE 
        assert without_atoms_no_roi.shape == DEFAULT_ABS_IMAGE_SHAPE 
        assert dark_no_roi.shape == DEFAULT_ABS_IMAGE_SHAPE

        expected_without_atoms_sum = np.prod(DEFAULT_ABS_IMAGE_SHAPE) * float(BASE_LIGHT_LEVEL + BASE_DARK_LEVEL) 
        expected_dark_sum = np.prod(DEFAULT_ABS_IMAGE_SHAPE) * float(BASE_DARK_LEVEL)
        #Not exact because of rounding
        expected_with_atoms_sum = (np.prod(DEFAULT_ABS_IMAGE_SHAPE) * float(BASE_LIGHT_LEVEL + BASE_DARK_LEVEL)
         - np.square(DEFAULT_ABS_SQUARE_WIDTH) * (1.0 - DEFAULT_ABSORPTION) * BASE_LIGHT_LEVEL)

        assert np.isclose(np.sum(without_atoms_no_roi.astype(float)), expected_without_atoms_sum)
        assert np.isclose(np.sum(dark_no_roi.astype(float)), expected_dark_sum)
        #Increased rtol to allow for rounding errors
        assert np.isclose(np.sum(with_atoms_no_roi.astype(float)), expected_with_atoms_sum, rtol = 1e-3)

        with_atoms_roi, without_atoms_roi, dark_roi = returned_array_roi   

        assert with_atoms_roi.dtype == np.ushort
        assert without_atoms_roi.dtype == np.ushort 
        assert dark_roi.dtype == np.ushort
        assert with_atoms_roi.shape == DEFAULT_ABSORPTION_IMAGE_ROI_SHAPE 
        assert without_atoms_roi.shape == DEFAULT_ABSORPTION_IMAGE_ROI_SHAPE 
        assert dark_roi.shape == DEFAULT_ABSORPTION_IMAGE_ROI_SHAPE

        expected_without_atoms_sum_roi = np.prod(DEFAULT_ABSORPTION_IMAGE_ROI_SHAPE) * float(BASE_LIGHT_LEVEL + BASE_DARK_LEVEL) 
        expected_dark_sum_roi = np.prod(DEFAULT_ABSORPTION_IMAGE_ROI_SHAPE) * float(BASE_DARK_LEVEL)
        #Not exact because of rounding
        expected_with_atoms_sum_roi = (np.prod(DEFAULT_ABSORPTION_IMAGE_ROI_SHAPE) * float(BASE_LIGHT_LEVEL + BASE_DARK_LEVEL)
         - np.square(DEFAULT_ABS_SQUARE_WIDTH) * (1.0 - DEFAULT_ABSORPTION) * BASE_LIGHT_LEVEL)

        assert np.isclose(np.sum(without_atoms_roi.astype(float)), expected_without_atoms_sum_roi)
        assert np.isclose(np.sum(dark_roi.astype(float)), expected_dark_sum_roi)
        #Increased rtol to allow for rounding errors
        assert np.isclose(np.sum(with_atoms_roi.astype(float)), expected_with_atoms_sum_roi, rtol = 1e-3)
    finally:
        shutil.rmtree(measurement_pathname)

def test_get_raw_pixels_na_catch():
    type_name = "na_catch"
    function_to_test = analysis_functions.get_raw_pixels_na_catch 
    _get_raw_pixels_test_helper(type_name, function_to_test)

def test_get_raw_pixels_side():
    type_name = "side_low_mag" 
    function_to_test = analysis_functions.get_raw_pixels_side 
    _get_raw_pixels_test_helper(type_name, function_to_test)


def test_get_raw_pixels_top_A():
    type_name = "top_double" 
    function_to_test = analysis_functions.get_raw_pixels_top_A
    _get_raw_pixels_test_helper(type_name, function_to_test)


def test_get_raw_pixels_top_B():
    type_name = "top_double" 
    function_to_test = analysis_functions.get_raw_pixels_top_B
    _get_raw_pixels_test_helper(type_name, function_to_test)


def _get_abs_image_test_helper(type_name, function_to_test):
    try:
        measurement_pathname, my_measurement, my_run = create_measurement(type_name, ROI = DEFAULT_ABSORPTION_IMAGE_ROI, 
                                                        norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        absorption_image = function_to_test(my_measurement, my_run)
        cropped_default_absorption_image = get_default_absorption_image(crop_to_roi = True)
        assert np.all(np.isclose(cropped_default_absorption_image, absorption_image, rtol = 1e-3))  
        #Try rebinning pixels
        REBIN_NUM = 2
        absorption_image_rebinned_pixels = function_to_test(my_measurement, my_run, rebin_pixel_num = REBIN_NUM)    
        assert absorption_image_rebinned_pixels.size == np.square((DEFAULT_ABSORPTION_IMAGE_ROI_SHAPE[0] - 1) // REBIN_NUM)
        assert np.isclose(np.sum(absorption_image) / 4, np.sum(absorption_image_rebinned_pixels), rtol = 3e-2)
    finally:
        shutil.rmtree(measurement_pathname)

def test_get_abs_image_na_catch():
    type_name = "na_catch"
    function_to_test = analysis_functions.get_abs_image_na_catch
    _get_abs_image_test_helper(type_name, function_to_test)

def test_get_abs_image_side():
    type_name = "side_low_mag"
    function_to_test = analysis_functions.get_abs_image_side
    _get_abs_image_test_helper(type_name, function_to_test)

def test_get_abs_image_top_A():
    type_name = "top_double"
    function_to_test = analysis_functions.get_abs_image_top_A
    _get_abs_image_test_helper(type_name, function_to_test)

def test_get_abs_image_top_B():
    type_name = "top_double" 
    function_to_test = analysis_functions.get_abs_image_top_B
    _get_abs_image_test_helper(type_name, function_to_test)

def test_get_abs_images_top_double():
    type_name = "top_double" 

    def abs_image_A_split_off(my_measurement, my_run, rebin_pixel_num = None):
        return analysis_functions.get_abs_images_top_double(my_measurement, my_run, rebin_pixel_num = rebin_pixel_num)[0] 
    
    def abs_image_B_split_off(my_measurement, my_run, rebin_pixel_num = None):
        return analysis_functions.get_abs_images_top_double(my_measurement, my_run, rebin_pixel_num = rebin_pixel_num)[1]
    
    _get_abs_image_test_helper(type_name, abs_image_A_split_off)
    _get_abs_image_test_helper(type_name, abs_image_B_split_off)

def _get_od_image_test_helper(type_name, function_to_test):
    try:
        measurement_pathname, my_measurement, my_run = create_measurement(type_name, ROI = DEFAULT_ABSORPTION_IMAGE_ROI, 
                                                                 norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        od_image = function_to_test(my_measurement, my_run) 
        cropped_default_od_image = -np.log(get_default_absorption_image(crop_to_roi = True))
        assert np.all(np.isclose(cropped_default_od_image, od_image, rtol = 1e-3))
        REBIN_NUM = 2
        od_image_rebinned_pixels = function_to_test(my_measurement, my_run, rebin_pixel_num = REBIN_NUM)    
        assert od_image_rebinned_pixels.size == np.square((DEFAULT_ABSORPTION_IMAGE_ROI_SHAPE[0] - 1) // REBIN_NUM)
        assert np.isclose(np.sum(od_image) / 4, np.sum(od_image_rebinned_pixels), rtol = 3e-2)    
    finally:
        shutil.rmtree(measurement_pathname)

def test_get_od_image_na_catch():
    type_name = "na_catch"
    function_to_test = analysis_functions.get_od_image_na_catch
    _get_od_image_test_helper(type_name, function_to_test) 

def test_get_od_image_side():
    type_name = "side_low_mag"
    function_to_test = analysis_functions.get_od_image_side 
    _get_od_image_test_helper(type_name, function_to_test)

def test_get_od_image_top_A():
    type_name = "top_double" 
    function_to_test = analysis_functions.get_od_image_top_A 
    _get_od_image_test_helper(type_name, function_to_test)

def test_get_od_image_top_B():
    type_name = "top_double" 
    function_to_test = analysis_functions.get_od_image_top_B 
    _get_od_image_test_helper(type_name, function_to_test)

def test_get_od_images_top_double():
    type_name = "top_double" 

    def od_image_A_split_off(my_measurement, my_run, rebin_pixel_num = None):
        return analysis_functions.get_od_images_top_double(my_measurement, my_run, rebin_pixel_num = rebin_pixel_num)[0] 
    
    def od_image_B_split_off(my_measurement, my_run, rebin_pixel_num = None):
        return analysis_functions.get_od_images_top_double(my_measurement, my_run, rebin_pixel_num = rebin_pixel_num)[1]
    
    _get_od_image_test_helper(type_name, od_image_A_split_off)
    _get_od_image_test_helper(type_name, od_image_B_split_off)


def _get_od_pixel_sum_test_helper(type_name, function_to_test):
    try:
        measurement_pathname, my_measurement, my_run = create_measurement(type_name, ROI = DEFAULT_ABSORPTION_IMAGE_ROI, 
                                                                          norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        pixel_sum = function_to_test(my_measurement, my_run) 
        expected_pixel_sum = np.square(DEFAULT_ABS_SQUARE_WIDTH)
        assert np.isclose(expected_pixel_sum, pixel_sum, rtol = 1e-3)
        #Once again, rebin
        REBIN_NUM = 2
        rebinned_pixel_sum = function_to_test(my_measurement, my_run, rebin_pixel_num = REBIN_NUM)
        assert np.isclose(pixel_sum / 4, rebinned_pixel_sum, rtol = 3e-2)
    finally:
        shutil.rmtree(measurement_pathname)


def test_get_od_pixel_sum_na_catch():
    type_name = "na_catch" 
    function_to_test = analysis_functions.get_od_pixel_sum_na_catch
    _get_od_pixel_sum_test_helper(type_name, function_to_test)

def test_get_od_pixel_sum_side():
    type_name = "side_low_mag" 
    function_to_test = analysis_functions.get_od_pixel_sum_side
    _get_od_pixel_sum_test_helper(type_name, function_to_test) 

def test_get_od_pixel_sum_top_A():
    type_name = "top_double" 
    function_to_test = analysis_functions.get_od_pixel_sum_top_A 
    _get_od_pixel_sum_test_helper(type_name, function_to_test)

def test_get_od_pixel_sum_top_B():
    type_name = "top_double" 
    function_to_test = analysis_functions.get_od_pixel_sum_top_A 
    _get_od_pixel_sum_test_helper(type_name, function_to_test)

def test_get_od_pixel_sums_top_double():
    type_name = "top_double" 

    def od_pixel_sum_A_split_off(my_measurement, my_run, rebin_pixel_num = None):
        return analysis_functions.get_od_pixel_sums_top_double(my_measurement, my_run, rebin_pixel_num = rebin_pixel_num)[0]
    
    def od_pixel_sum_B_split_off(my_measurement, my_run, rebin_pixel_num = None):
        return analysis_functions.get_od_pixel_sums_top_double(my_measurement, my_run, rebin_pixel_num = rebin_pixel_num)[1]
    
    _get_od_pixel_sum_test_helper(type_name, od_pixel_sum_A_split_off)
    _get_od_pixel_sum_test_helper(type_name, od_pixel_sum_B_split_off)

li_6_res_cross_section = image_processing_functions._get_res_cross_section_from_species("6Li")
li_6_linewidth = image_processing_functions._get_linewidth_from_species("6Li")

def _get_atom_density_test_helper(type_name, function_to_test, cross_section, experiment_param_values = None, 
                                           run_param_values = None, fun_kwargs = None):
    if fun_kwargs is None:
        fun_kwargs = {}
    try:
        measurement_pathname, my_measurement, my_run = create_measurement(type_name, ROI = DEFAULT_ABSORPTION_IMAGE_ROI, 
                                                                          norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX,
                                                                        experiment_param_values = experiment_param_values, 
                                                                        run_param_values = run_param_values)
        atom_densities = function_to_test(my_measurement, my_run, **fun_kwargs)
        cropped_expected_densities = -np.log(get_default_absorption_image(crop_to_roi = True)) / cross_section 
        assert np.all(np.isclose(atom_densities, cropped_expected_densities, rtol = 1e-3))
        REBIN_NUM = 2
        fun_kwargs["rebin_pixel_num"] = REBIN_NUM
        atom_densities_rebinned_pixels = function_to_test(my_measurement, my_run, **fun_kwargs)
        assert atom_densities_rebinned_pixels.size == np.square((DEFAULT_ABSORPTION_IMAGE_ROI_SHAPE[0] - 1) // REBIN_NUM)
        assert np.isclose(np.sum(atom_densities) / 4, np.sum(atom_densities_rebinned_pixels), rtol = 3e-2)
    finally:
        shutil.rmtree(measurement_pathname)

def test_get_atom_density_side_li_lf():
    experiment_param_values = {
        "li_lf_freq_multiplier":L33T_DUMMY,
        "li_lf_res_freq":PI_DUMMY,
        "li_side_sigma_multiplier":E_DUMMY
    }
    run_param_values_resonant = {
        "LFImgFreq":PI_DUMMY
    }
    run_param_values_detuned = {
        "LFImgFreq":li_6_linewidth / L33T_DUMMY + PI_DUMMY
    }
    dummy_rescaled_cross_section = li_6_res_cross_section * E_DUMMY
    _get_atom_density_test_helper("side_low_mag", analysis_functions.get_atom_density_side_li_lf, dummy_rescaled_cross_section, 
                                           experiment_param_values = experiment_param_values, run_param_values = run_param_values_resonant)
    detuned_effective_cross_section = dummy_rescaled_cross_section / 5
    _get_atom_density_test_helper("side_low_mag", analysis_functions.get_atom_density_side_li_lf, detuned_effective_cross_section, 
                                  experiment_param_values = experiment_param_values, run_param_values = run_param_values_detuned)

def _get_hf_atom_density_test_helper(measurement_type, function_to_test, run_param_keys):

    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": L33T_DUMMY,
        "state_2_unitarity_res_freq_MHz":E_DUMMY,
        "state_3_unitarity_res_freq_MHz":PI_DUMMY,
        "hf_lock_unitarity_resonance_value":SQRT_2_DUMMY,
        "hf_lock_rr_resonance_value":SQRT_3_DUMMY,
        "hf_lock_zero_crossing_resonance_value":SQRT_5_DUMMY,
        "hf_lock_setpoint":SQRT_7_DUMMY,
        "hf_lock_frequency_multiplier":SQRT_10_DUMMY,
        "li_side_sigma_multiplier":SQRT_11_DUMMY, 
        "li_top_sigma_multiplier":SQRT_14_DUMMY,
        "li_hf_freq_multiplier":SQRT_13_DUMMY
    }

    on_res_unitarity_nominal_frequency_1 = (
        hf_atom_density_experiment_param_values["state_1_unitarity_res_freq_MHz"] - 
        hf_atom_density_experiment_param_values["hf_lock_frequency_multiplier"]/hf_atom_density_experiment_param_values["li_hf_freq_multiplier"] * 
        (hf_atom_density_experiment_param_values["hf_lock_setpoint"] - hf_atom_density_experiment_param_values["hf_lock_unitarity_resonance_value"])
    )
    on_res_unitarity_nominal_frequency_2 = (
        hf_atom_density_experiment_param_values["state_2_unitarity_res_freq_MHz"] - 
        hf_atom_density_experiment_param_values["hf_lock_frequency_multiplier"]/hf_atom_density_experiment_param_values["li_hf_freq_multiplier"] * 
        (hf_atom_density_experiment_param_values["hf_lock_setpoint"] - hf_atom_density_experiment_param_values["hf_lock_unitarity_resonance_value"]) 
    )
    on_res_unitarity_nominal_frequency_3 = (
        hf_atom_density_experiment_param_values["state_3_unitarity_res_freq_MHz"] - 
        hf_atom_density_experiment_param_values["hf_lock_frequency_multiplier"]/hf_atom_density_experiment_param_values["li_hf_freq_multiplier"] * 
        (hf_atom_density_experiment_param_values["hf_lock_setpoint"] - hf_atom_density_experiment_param_values["hf_lock_unitarity_resonance_value"]) 
    )

    on_res_rapid_ramp_nominal_frequency_1 = (
        hf_atom_density_experiment_param_values["state_1_unitarity_res_freq_MHz"] - 
        hf_atom_density_experiment_param_values["hf_lock_frequency_multiplier"]/hf_atom_density_experiment_param_values["li_hf_freq_multiplier"] * 
        (hf_atom_density_experiment_param_values["hf_lock_setpoint"] - hf_atom_density_experiment_param_values["hf_lock_rr_resonance_value"])
    )

    on_res_zero_crossing_nominal_frequency_1 = (
        hf_atom_density_experiment_param_values["state_1_unitarity_res_freq_MHz"] - 
        hf_atom_density_experiment_param_values["hf_lock_frequency_multiplier"]/hf_atom_density_experiment_param_values["li_hf_freq_multiplier"] * 
        (hf_atom_density_experiment_param_values["hf_lock_setpoint"] - hf_atom_density_experiment_param_values["hf_lock_zero_crossing_resonance_value"])
    )

    off_res_unitarity_nominal_frequency_1 = on_res_unitarity_nominal_frequency_1 + li_6_linewidth / hf_atom_density_experiment_param_values["li_hf_freq_multiplier"]



    if measurement_type == "side_high_mag":
        dummy_rescaled_cross_section = li_6_res_cross_section * hf_atom_density_experiment_param_values["li_side_sigma_multiplier"]
    elif measurement_type == "top_double":
        dummy_rescaled_cross_section = li_6_res_cross_section * hf_atom_density_experiment_param_values["li_top_sigma_multiplier"]


    run_param_values_on_res_unitarity_1 = {} 
    run_param_values_on_res_unitarity_2 = {} 
    run_param_values_on_res_unitarity_3 = {} 
    run_param_values_on_res_zero_crossing_1 = {} 
    run_param_values_on_res_rapid_ramp_1 = {}
    run_param_values_off_res_unitarity_1 = {}
    for run_param_key in run_param_keys:
        run_param_values_on_res_unitarity_1[run_param_key] = on_res_unitarity_nominal_frequency_1
        run_param_values_on_res_unitarity_2[run_param_key] = on_res_unitarity_nominal_frequency_2
        run_param_values_on_res_unitarity_3[run_param_key] = on_res_unitarity_nominal_frequency_3
        run_param_values_on_res_zero_crossing_1[run_param_key] = on_res_zero_crossing_nominal_frequency_1
        run_param_values_on_res_rapid_ramp_1[run_param_key] = on_res_rapid_ramp_nominal_frequency_1
        run_param_values_off_res_unitarity_1[run_param_key] = off_res_unitarity_nominal_frequency_1

    #Test getting different states
    _get_atom_density_test_helper(measurement_type, function_to_test, dummy_rescaled_cross_section, 
                                  experiment_param_values = hf_atom_density_experiment_param_values, run_param_values = run_param_values_on_res_unitarity_1, 
                                  fun_kwargs = {"state_index":1})
    _get_atom_density_test_helper(measurement_type, function_to_test, dummy_rescaled_cross_section, 
                                  experiment_param_values = hf_atom_density_experiment_param_values, run_param_values = run_param_values_on_res_unitarity_2, 
                                  fun_kwargs = {"state_index":2})
    _get_atom_density_test_helper(measurement_type, function_to_test, dummy_rescaled_cross_section, 
                                  experiment_param_values = hf_atom_density_experiment_param_values, run_param_values = run_param_values_on_res_unitarity_3, 
                                  fun_kwargs = {"state_index":3})
    
    #Test off-resonant imaging 

    off_res_dummy_rescaled_cross_section = dummy_rescaled_cross_section / 5
    _get_atom_density_test_helper(measurement_type, function_to_test, off_res_dummy_rescaled_cross_section, 
                                  experiment_param_values = hf_atom_density_experiment_param_values, run_param_values = run_param_values_off_res_unitarity_1, 
                                  fun_kwargs = {"state_index":1})
    

    #Test different field conditions 
    _get_atom_density_test_helper(measurement_type, function_to_test, dummy_rescaled_cross_section, 
                                  experiment_param_values = hf_atom_density_experiment_param_values, run_param_values = run_param_values_on_res_rapid_ramp_1, 
                                  fun_kwargs = {"state_index":1, "b_field_condition":"rapid_ramp"})    


    _get_atom_density_test_helper(measurement_type, function_to_test, dummy_rescaled_cross_section, 
                                  experiment_param_values = hf_atom_density_experiment_param_values, run_param_values = run_param_values_on_res_zero_crossing_1, 
                                  fun_kwargs = {"state_index":1, "b_field_condition":"zero_crossing"})    
        
def test_get_atom_density_side_li_hf():
    _get_hf_atom_density_test_helper("side_high_mag", analysis_functions.get_atom_density_side_li_hf, ("ImagFreq0",))

def test_get_atom_density_top_A_abs():
    _get_hf_atom_density_test_helper("top_double", analysis_functions.get_atom_density_top_A_abs, ("ImagFreq1",))

def test_get_atom_density_top_B_abs():
    _get_hf_atom_density_test_helper("top_double", analysis_functions.get_atom_density_top_B_abs, ("ImagFreq2",))

def test_get_atom_densities_top_abs():
    #analysis_functions.get_atom_densities_top_abs is only a thin wrapper around 
    #get_atom_density_top_A_abs and get_atom_density_top_B_abs; accordingly, we 
    #just make sure the piping is being done correctly.

    def top_abs_A_split_off(my_measurement, my_run, **fun_kwargs):
        fun_kwargs["first_state_index"] = fun_kwargs.pop("state_index")
        return analysis_functions.get_atom_densities_top_abs(my_measurement, my_run, **fun_kwargs)[0] 
    
    def top_abs_B_split_off(my_measurement, my_run, **fun_kwargs):
        fun_kwargs["second_state_index"] = fun_kwargs.pop("state_index")
        return analysis_functions.get_atom_densities_top_abs(my_measurement, my_run, **fun_kwargs)[1] 
    
    _get_hf_atom_density_test_helper("top_double", top_abs_A_split_off, ("ImagFreq1", "ImagFreq2"))
    _get_hf_atom_density_test_helper("top_double", top_abs_B_split_off, ("ImagFreq1", "ImagFreq2"))
    
def test_get_atom_densities_top_polrot():
    experiment_param_values_polrot = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_2_unitarity_res_freq_MHz":10.0,
        "state_3_unitarity_res_freq_MHz":20.0,
        "hf_lock_unitarity_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_side_sigma_multiplier":1.0, 
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "top_um_per_pixel":SQRT_2_DUMMY,
        "polrot_phase_sign":1.0, 
        "top_camera_quantum_efficiency":0.7,
        "top_camera_counts_per_photoelectron":2.5,
        "top_camera_post_atom_photon_transmission":0.8,
        "top_camera_saturation_ramsey_fudge":1.1
    }
    run_param_values_polrot = {
        "ImagFreq1":3.0, 
        "ImagFreq2":23.0,
        "ImageTime":500.0
    }
    default_absorption_image = get_default_absorption_image()
    default_image_stack = generate_image_stack_from_absorption(default_absorption_image)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = default_image_stack,
                                                        run_param_values = run_param_values_polrot, experiment_param_values = experiment_param_values_polrot, 
                                                        ROI = DEFAULT_ABSORPTION_IMAGE_ROI, norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        #Manually call the polrot function to find what the two returned densities ought to be given the absorption
        #Set up the frequencies...
        hf_lock_multiplier = experiment_param_values_polrot["hf_lock_frequency_multiplier"]
        hf_lock_offset = (experiment_param_values_polrot["hf_lock_setpoint"] - experiment_param_values_polrot["hf_lock_unitarity_resonance_value"]) * hf_lock_multiplier
        imaging_multiplier = experiment_param_values_polrot["li_hf_freq_multiplier"]
        frequency_A = run_param_values_polrot["ImagFreq1"] 
        frequency_B = run_param_values_polrot["ImagFreq2"]
        resonance_1 = experiment_param_values_polrot["state_1_unitarity_res_freq_MHz"]
        resonance_2 = experiment_param_values_polrot["state_3_unitarity_res_freq_MHz"]
        detuning_1A = imaging_multiplier * (frequency_A - resonance_1) + hf_lock_offset
        detuning_1B = imaging_multiplier * (frequency_B - resonance_1) + hf_lock_offset
        detuning_2A = imaging_multiplier * (frequency_A - resonance_2) + hf_lock_offset 
        detuning_2B = imaging_multiplier * (frequency_B - resonance_2) + hf_lock_offset 
        #First do it without saturation 
        single_pixel_abs_array = np.array([1.0 / np.e])
        single_pixel_polrot_return_arrays = image_processing_functions.get_atom_density_from_polrot_images(
            single_pixel_abs_array, single_pixel_abs_array, 
            detuning_1A, detuning_1B, detuning_2A, detuning_2B, 
            phase_sign = experiment_param_values_polrot["polrot_phase_sign"]
        )
        single_pixel_polrot_density_1, single_pixel_polrot_density_2 = single_pixel_polrot_return_arrays
        expected_polrot_density_1 = generate_default_image_density_pattern(crop_to_roi = True, density_value = single_pixel_polrot_density_1[0])
        expected_polrot_density_2 = generate_default_image_density_pattern(crop_to_roi = True, density_value = single_pixel_polrot_density_2[0])
        polrot_density_1, polrot_density_2 = analysis_functions.get_atom_densities_top_polrot(my_measurement, my_run, first_state_index = 1, 
                                                                                              second_state_index = 3, b_field_condition = "unitarity", 
                                                                                              use_saturation = False)
        assert np.all(np.isclose(polrot_density_1, expected_polrot_density_1, rtol = 1e-3))
        assert np.all(np.isclose(polrot_density_2, expected_polrot_density_2, rtol = 1e-3))
        #Test that pixel rebinning works
        REBIN_NUM = 2
        polrot_density_1_rebinned_pixels, polrot_density_2_rebinned_pixels = analysis_functions.get_atom_densities_top_polrot(
                                                                            my_measurement, my_run, first_state_index = 1,
                                                                            second_state_index = 3, rebin_pixel_num = REBIN_NUM,
                                                                            b_field_condition = "unitarity",
                                                                            use_saturation = False)
        assert polrot_density_1_rebinned_pixels.size == np.square((DEFAULT_ABSORPTION_IMAGE_ROI_SHAPE[0] - 1) // REBIN_NUM)
        assert np.isclose(np.sum(polrot_density_1) / 4, np.sum(polrot_density_1_rebinned_pixels), rtol = 1e-1)
        #Then do it with saturation
        saturation_counts = analysis_functions.get_saturation_counts_top(my_measurement, my_run, apply_ramsey_fudge = True)
        intensity_A = BASE_LIGHT_LEVEL
        intensity_B = BASE_LIGHT_LEVEL
        single_pixel_polrot_return_arrays_saturated = image_processing_functions.get_atom_density_from_polrot_images(
            single_pixel_abs_array, single_pixel_abs_array, 
            detuning_1A, detuning_1B, detuning_2A, detuning_2B, 
            phase_sign = experiment_param_values_polrot["polrot_phase_sign"],
            intensities_A = intensity_A, intensities_B = intensity_B, 
            intensities_sat = saturation_counts
        )
        single_pixel_polrot_density_1_saturated, single_pixel_polrot_density_2_saturated = single_pixel_polrot_return_arrays_saturated
        expected_polrot_density_1_saturated = generate_default_image_density_pattern(crop_to_roi = True, density_value = single_pixel_polrot_density_1_saturated[0])
        expected_polrot_density_2_saturated = generate_default_image_density_pattern(crop_to_roi = True, density_value = single_pixel_polrot_density_2_saturated[0])
        polrot_density_1_saturated, polrot_density_2_saturated = analysis_functions.get_atom_densities_top_polrot(my_measurement, my_run, 
                                                                                first_state_index = 1, second_state_index = 3, b_field_condition = "unitarity", 
                                                                                use_saturation = True)
        assert np.all(np.isclose(polrot_density_1_saturated, expected_polrot_density_1_saturated, rtol = 1e-3))
        assert np.all(np.isclose(polrot_density_2_saturated, expected_polrot_density_2_saturated, rtol = 1e-3))
    finally:
        shutil.rmtree(measurement_pathname)


EXPECTED_AUTOCUT_FREE_CROP = (192, 193, 320, 320)
EXPECTED_AUTOCUT_FIXED_CROP = (194, 196, 318, 316)
EXPECTED_AUTOCUT_FREE_HEIGHT = DEFAULT_ABS_SQUARE_WIDTH
EXPECTED_AUTOCUT_FREE_WIDTH = DEFAULT_ABS_SQUARE_WIDTH + 1
ENFORCED_AUTOCUT_FIXED_HEIGHT = DEFAULT_ABS_SQUARE_WIDTH - 7 
ENFORCED_AUTOCUT_FIXED_WIDTH = DEFAULT_ABS_SQUARE_WIDTH - 3

def test_box_autocut():
    experiment_param_values = {
        "axicon_diameter_pix":ENFORCED_AUTOCUT_FIXED_WIDTH,
        "box_length_pix":ENFORCED_AUTOCUT_FIXED_HEIGHT
    }
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", experiment_param_values = experiment_param_values)
        box_autocut_image = get_box_autocut_absorption_image(crop_to_roi = False)
        box_autocut_densities = -np.log(box_autocut_image)
        autocut_box_crop_widths_free = analysis_functions.box_autocut(my_measurement, box_autocut_densities, widths_free = True)
        xmin_free, ymin_free, xmax_free, ymax_free = autocut_box_crop_widths_free
        autocut_width_free = xmax_free - xmin_free
        autocut_height_free = ymax_free - ymin_free
        #Radius is rounded to nearest integer, hence width is even
        assert autocut_width_free == DEFAULT_ABS_SQUARE_WIDTH + 1
        assert autocut_height_free == DEFAULT_ABS_SQUARE_WIDTH
        assert autocut_box_crop_widths_free == EXPECTED_AUTOCUT_FREE_CROP
        #Now test the width with constrained sizes
        autocut_box_crop_widths_fixed = analysis_functions.box_autocut(my_measurement, box_autocut_densities, widths_free = False)
        xmin_fixed, ymin_fixed, xmax_fixed, ymax_fixed = autocut_box_crop_widths_fixed 
        autocut_width_fixed = xmax_fixed - xmin_fixed 
        assert autocut_width_fixed == experiment_param_values["axicon_diameter_pix"]
        autocut_height_fixed = ymax_fixed - ymin_fixed
        assert autocut_height_fixed == experiment_param_values["box_length_pix"]
        assert autocut_box_crop_widths_fixed == EXPECTED_AUTOCUT_FIXED_CROP
    finally:
        shutil.rmtree(measurement_pathname)

def test_get_atom_densities_box_autocut():
    TEST_AXICON_TILT_DEG = 7.5
    SUPERSAMPLE_SCALE_FACTOR = 2
    box_autocut_image = get_box_autocut_absorption_image() 
    tilted_box_autocut_image = scipy.ndimage.rotate(box_autocut_image, -TEST_AXICON_TILT_DEG, reshape = False)
    with_tilt_image_stack = generate_image_stack_from_absorption(tilted_box_autocut_image) 
    no_tilt_image_stack = generate_image_stack_from_absorption(box_autocut_image)
    hf_atom_density_experiment_param_values_with_tilt = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0,
        "hf_lock_setpoint":0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0, 
        "axicon_diameter_pix":ENFORCED_AUTOCUT_FIXED_WIDTH,
        "box_length_pix":ENFORCED_AUTOCUT_FIXED_HEIGHT,
        "axicon_tilt_deg":TEST_AXICON_TILT_DEG
    }
    hf_atom_density_experiment_param_values_no_tilt = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0,
        "hf_lock_setpoint":0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0, 
        "axicon_diameter_pix":ENFORCED_AUTOCUT_FIXED_WIDTH,
        "box_length_pix":ENFORCED_AUTOCUT_FIXED_HEIGHT,
        "axicon_tilt_deg":0.0
    }
    run_param_values = {
        "ImagFreq1":0.0, 
        "ImagFreq2":0.0
    }
    autocut_fun_kwargs_no_stored_density = {
        "imaging_mode":"abs", 
        "widths_free":True
    }

    autocut_fun_kwargs_fixed_width = {
        "imaging_mode":"abs",
        "widths_free":False
    }
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = with_tilt_image_stack, ROI = DEFAULT_ABSORPTION_IMAGE_ROI, 
                                            norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX, run_param_values = run_param_values, 
                                            experiment_param_values = hf_atom_density_experiment_param_values_with_tilt)
        #First try not pre-processing the density
        autocut_density_1, autocut_density_2 = analysis_functions.get_atom_densities_box_autocut(my_measurement, my_run, **autocut_fun_kwargs_no_stored_density, 
                                                                                                 supersample_scale_factor = SUPERSAMPLE_SCALE_FACTOR)
        autocut_density_free_height, autocut_density_free_width = autocut_density_1.shape
        expected_free_height_with_supersample = EXPECTED_AUTOCUT_FREE_HEIGHT * SUPERSAMPLE_SCALE_FACTOR
        assert autocut_density_free_height == expected_free_height_with_supersample
        #Ad hoc... I don't want to figure out the off by 1
        expected_free_width_with_supersample = EXPECTED_AUTOCUT_FREE_WIDTH * SUPERSAMPLE_SCALE_FACTOR - 1
        assert autocut_density_free_width == expected_free_width_with_supersample

        #With width fixed
        autocut_density_1_fixed, _ = analysis_functions.get_atom_densities_box_autocut(my_measurement, my_run, **autocut_fun_kwargs_fixed_width, 
                                                                                       supersample_scale_factor = SUPERSAMPLE_SCALE_FACTOR)
        autocut_density_fixed_height, autocut_density_fixed_width = autocut_density_1_fixed.shape
        enforced_autocut_fixed_height_with_supersample = ENFORCED_AUTOCUT_FIXED_HEIGHT * SUPERSAMPLE_SCALE_FACTOR
        assert autocut_density_fixed_height == enforced_autocut_fixed_height_with_supersample
        enforced_autocut_fixed_width_with_supersample = ENFORCED_AUTOCUT_FIXED_WIDTH * SUPERSAMPLE_SCALE_FACTOR
        assert autocut_density_fixed_width == enforced_autocut_fixed_width_with_supersample
    finally:
        shutil.rmtree(measurement_pathname)

    #And now we try it with an un-rotated image 

    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = no_tilt_image_stack, ROI = DEFAULT_ABSORPTION_IMAGE_ROI, 
                                            norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX, run_param_values = run_param_values, 
                                            experiment_param_values = hf_atom_density_experiment_param_values_no_tilt)
        #First try not pre-processing the density
        autocut_density_1, autocut_density_2 = analysis_functions.get_atom_densities_box_autocut(my_measurement, my_run, **autocut_fun_kwargs_no_stored_density, 
                                                                                                 supersample_and_rotate = False)
        autocut_density_free_height, autocut_density_free_width = autocut_density_1.shape
        assert autocut_density_free_height == EXPECTED_AUTOCUT_FREE_HEIGHT
        assert autocut_density_free_width == EXPECTED_AUTOCUT_FREE_WIDTH

        #With width fixed
        autocut_density_1_fixed, _ = analysis_functions.get_atom_densities_box_autocut(my_measurement, my_run, **autocut_fun_kwargs_fixed_width, 
                                                                                       supersample_and_rotate = False)
        autocut_density_fixed_height, autocut_density_fixed_width = autocut_density_1_fixed.shape
        assert autocut_density_fixed_height == ENFORCED_AUTOCUT_FIXED_HEIGHT
        assert autocut_density_fixed_width == ENFORCED_AUTOCUT_FIXED_WIDTH
    finally:
        shutil.rmtree(measurement_pathname)



def test_get_top_atom_densities_COM_centered():
    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0,
        "hf_lock_setpoint":0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0, 
        "top_um_per_pixel":E_DUMMY
    }
    run_param_values = {
        "ImagFreq1":0.0, 
        "ImagFreq2":0.0
    }
    no_stored_density_kwargs = {
        "imaging_mode":"abs"
    }
    offset_box_image = get_offset_box_absorption_image()
    offset_box_image_stack = generate_image_stack_from_absorption(offset_box_image)
    roi_cropped_offset_box_image = get_offset_box_absorption_image(crop_to_roi = True)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = offset_box_image_stack, 
                                                        run_param_values = run_param_values, experiment_param_values = hf_atom_density_experiment_param_values, 
                                                        ROI = DEFAULT_ABSORPTION_IMAGE_ROI, norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        #First test cropping 
        cropped_density_1, cropped_density_2 = analysis_functions.get_top_atom_densities_COM_centered(my_measurement, my_run, crop_not_pad = True,
                                                                                                  **no_stored_density_kwargs)
        cropped_density_ycom, cropped_density_xcom = image_processing_functions.get_image_coms(cropped_density_2)
        cropped_density_y_length = cropped_density_2.shape[0]
        cropped_density_x_length = cropped_density_2.shape[1]
        assert roi_cropped_offset_box_image.shape[0] == cropped_density_y_length + 2 * DENSITY_COM_OFFSET 
        assert roi_cropped_offset_box_image.shape[1] == cropped_density_x_length + 2 * DENSITY_COM_OFFSET
        cropped_density_y_center = (cropped_density_y_length - 1) / 2 
        cropped_density_x_center = (cropped_density_x_length - 1) / 2 
        assert np.isclose(cropped_density_x_center, cropped_density_xcom)
        assert np.isclose(cropped_density_y_center, cropped_density_ycom)
        #Now test padding 
        padded_density_1, padded_density_2 = analysis_functions.get_top_atom_densities_COM_centered(my_measurement, my_run, crop_not_pad = False, 
                                                                                               **no_stored_density_kwargs)
        padded_density_ycom, padded_density_xcom = image_processing_functions.get_image_coms(padded_density_2)
        padded_density_y_length = padded_density_2.shape[0]
        padded_density_x_length = padded_density_2.shape[1]
        assert roi_cropped_offset_box_image.shape[0] == padded_density_y_length - 2 * DENSITY_COM_OFFSET
        assert roi_cropped_offset_box_image.shape[1] == padded_density_x_length - 2 * DENSITY_COM_OFFSET
        padded_density_y_center = (padded_density_y_length - 1) / 2 
        padded_density_x_center = (padded_density_x_length - 1) / 2
        assert np.isclose(padded_density_y_center, padded_density_ycom) 
        assert np.isclose(padded_density_x_center, padded_density_xcom)
    finally:
        shutil.rmtree(measurement_pathname)

def test_get_top_atom_densities_supersampled_and_rotated():
    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0,
        "hf_lock_setpoint":0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0, 
        "top_um_per_pixel":E_DUMMY
    }
    run_param_values = {
        "ImagFreq1":0.0, 
        "ImagFreq2":0.0
    }
    no_stored_density_kwargs = {
        "imaging_mode":"abs"
    }
    TEST_ROTATION_ANGLE = 10.0
    rotated_rectangle_image = get_rotated_rectangle_absorption_image(-TEST_ROTATION_ANGLE)
    rotated_rectangle_image_stack = generate_image_stack_from_absorption(rotated_rectangle_image)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = rotated_rectangle_image_stack, 
                                                        run_param_values = run_param_values, experiment_param_values = hf_atom_density_experiment_param_values, 
                                                        ROI = DEFAULT_ABSORPTION_IMAGE_ROI, norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        #First without stored densities, and with clip disabled...
        supersampled_rotated_densities_1, supersampled_rotated_densities_2 = analysis_functions.get_top_atom_densities_supersampled_and_rotated(
                                            my_measurement, my_run, angle = TEST_ROTATION_ANGLE, **no_stored_density_kwargs, clip = False)
        assert np.all(supersampled_rotated_densities_1.shape == 2 * np.array(DEFAULT_ABSORPTION_IMAGE_ROI_SHAPE))
        assert np.allclose(supersampled_rotated_densities_1, supersampled_rotated_densities_2)
        #Check that the angle is correct 
        extracted_image_angle = image_processing_functions.get_image_principal_rotation_angle(
            supersampled_rotated_densities_1
        )
        assert np.isclose(extracted_image_angle, 0.0, atol = 1e-4)
    finally:
        shutil.rmtree(measurement_pathname)

    x_indices, y_indices = np.indices(rotated_rectangle_image.shape)
    *_, default_norm_xmax, default_norm_ymax = DEFAULT_ABSORPTION_IMAGE_NORM_BOX
    background_offset_multiplier = np.where(
        np.logical_and(
            x_indices > default_norm_xmax, 
            y_indices > default_norm_ymax
        ),
        np.exp(-1.0),
        1.0
    )
    background_offset_rotated_rectangle_image = rotated_rectangle_image * background_offset_multiplier
    background_offset_rotated_rectangle_image_stack = generate_image_stack_from_absorption(background_offset_rotated_rectangle_image)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = background_offset_rotated_rectangle_image_stack, 
                                                        run_param_values = run_param_values, experiment_param_values = hf_atom_density_experiment_param_values, 
                                                        ROI = DEFAULT_ABSORPTION_IMAGE_ROI, norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        #Now we use the cut
        supersampled_rotated_densities_1_clipped, supersampled_rotated_densities_2_clipped = analysis_functions.get_top_atom_densities_supersampled_and_rotated(
                                            my_measurement, my_run, angle = TEST_ROTATION_ANGLE, **no_stored_density_kwargs, clip = True)
        assert np.allclose(supersampled_rotated_densities_1_clipped, supersampled_rotated_densities_2_clipped)
        extracted_clipped_image_angle = image_processing_functions.get_image_principal_rotation_angle(
            supersampled_rotated_densities_1_clipped
        )
        assert np.isclose(extracted_clipped_image_angle, 0.0, atol = 3e-3)
        #Check that there's no residual zero values - which would indicate an inappropriate clip... 
        assert np.all(supersampled_rotated_densities_1_clipped != 0.0)
    finally:
        shutil.rmtree(measurement_pathname)

        




def _get_integrated_densities_test_helper(function_to_use, integration_axis):
    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0,
        "hf_lock_setpoint":0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0, 
        "top_um_per_pixel":E_DUMMY
    }
    run_param_values = {
        "ImagFreq1":0.0, 
        "ImagFreq2":0.0
    }
    no_stored_density_kwargs = {
        "imaging_mode":"abs"
    }
    stored_density_kwargs = {
        "first_stored_density_name":"densities_1", 
        "second_stored_density_name":"densities_3"
    }
    box_autocut_image = get_box_autocut_absorption_image()
    cropped_box_autocut_image = get_box_autocut_absorption_image(crop_to_roi = True)
    box_autocut_image_stack = generate_image_stack_from_absorption(box_autocut_image)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = box_autocut_image_stack, 
                                                        run_param_values = run_param_values, experiment_param_values = hf_atom_density_experiment_param_values, 
                                                        ROI = DEFAULT_ABSORPTION_IMAGE_ROI, norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        #Get the expected atom densities
        expected_atom_density = -np.log(cropped_box_autocut_image) / li_6_res_cross_section
        expected_integrated_density = np.sum(expected_atom_density, axis = integration_axis) * hf_atom_density_experiment_param_values["top_um_per_pixel"] 
        #First, don't store density
        integrated_densities = function_to_use(my_measurement, my_run, **no_stored_density_kwargs)
        integrated_density_1, _ = integrated_densities
        assert np.all(np.isclose(integrated_density_1, expected_integrated_density, rtol = 1e-3))
        #Then, do store density
        my_measurement.analyze_runs(analysis_functions.get_atom_densities_top_abs, ("densities_1", "densities_3"))
        integrated_densities_stored_density = function_to_use(my_measurement, my_run, **stored_density_kwargs)
        integrated_density_1_stored_density, _ = integrated_densities_stored_density
        assert np.all(np.isclose(expected_integrated_density, integrated_density_1_stored_density, rtol = 1e-3))
    finally:
        shutil.rmtree(measurement_pathname)

def test_get_x_integrated_atom_densities_top_double():
    _get_integrated_densities_test_helper(analysis_functions.get_x_integrated_atom_densities_top_double, 1)

def test_get_y_integrated_atom_densities_top_double():
    _get_integrated_densities_test_helper(analysis_functions.get_y_integrated_atom_densities_top_double, 0) 


def test_get_xy_atom_density_pixel_coms_top_double():
    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0,
        "hf_lock_setpoint":0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0, 
        "top_um_per_pixel":E_DUMMY
    }
    run_param_values = {
        "ImagFreq1":0.0, 
        "ImagFreq2":0.0
    }
    no_stored_density_kwargs = {
        "imaging_mode":"abs"
    }
    box_autocut_image = get_box_autocut_absorption_image()
    cropped_box_autocut_image = get_box_autocut_absorption_image(crop_to_roi = True)
    box_autocut_image_stack = generate_image_stack_from_absorption(box_autocut_image)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = box_autocut_image_stack, 
                                                        run_param_values = run_param_values, experiment_param_values = hf_atom_density_experiment_param_values, 
                                                        ROI = DEFAULT_ABSORPTION_IMAGE_ROI, norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        #Box is symmetric, so expected COM is in the middle
        cropped_image_dimensions = cropped_box_autocut_image.shape 
        cropped_image_y_len, cropped_image_x_len = cropped_image_dimensions 
        expected_y_com = (cropped_image_y_len - 1) / 2 
        expected_x_com = (cropped_image_x_len - 1) / 2
        #Get the expected atom densities
        x_com_1, y_com_1, x_com_2, y_com_2 = analysis_functions.get_xy_atom_density_pixel_coms_top_double(my_measurement, my_run, **no_stored_density_kwargs)
        #Then, do store density
        assert np.isclose(expected_x_com, x_com_1)
        assert np.isclose(expected_x_com, x_com_2) 
        assert np.isclose(expected_y_com, y_com_1) 
        assert np.isclose(expected_y_com, y_com_2)
    finally:
        shutil.rmtree(measurement_pathname)


def _get_atom_counts_test_helper(type_name, function_to_test, experiment_param_values = None, 
                                           run_param_values = None, fun_kwargs = None):
    if fun_kwargs is None:
        fun_kwargs = {}
    try:
        measurement_pathname, my_measurement, my_run = create_measurement(type_name, ROI = DEFAULT_ABSORPTION_IMAGE_ROI, 
                                                                          norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX,
                                                                        experiment_param_values = experiment_param_values, 
                                                                        run_param_values = run_param_values)
        atom_counts = function_to_test(my_measurement, my_run, **fun_kwargs)
        cropped_expected_densities = -np.log(get_default_absorption_image(crop_to_roi = True)) / li_6_res_cross_section 
        if type_name == "side_low_mag":
            pixel_length = my_measurement.experiment_parameters["side_low_mag_um_per_pixel"] 
        elif type_name == "side_high_mag":
            pixel_length = my_measurement.experiment_parameters["side_high_mag_um_per_pixel"]
        elif type_name == "top_double":
            pixel_length = my_measurement.experiment_parameters["top_um_per_pixel"]
        expected_counts = np.square(pixel_length) * np.sum(cropped_expected_densities)
        assert np.isclose(atom_counts, expected_counts, rtol = 1e-3)
    finally:
        shutil.rmtree(measurement_pathname)

def test_get_atom_count_side_li_lf():
    experiment_param_values = {
        "li_lf_freq_multiplier":1,
        "li_lf_res_freq":0.0,
        "li_side_sigma_multiplier":1.0,
        "side_low_mag_um_per_pixel":E_DUMMY
    }
    run_param_values = {
        "LFImgFreq":0.0
    }
    _get_atom_counts_test_helper("side_low_mag", analysis_functions.get_atom_count_side_li_lf, 
                        experiment_param_values = experiment_param_values, run_param_values = run_param_values)
    
def test_get_atom_count_side_li_hf():
    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_2_unitarity_res_freq_MHz":0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0.0,
        "hf_lock_rr_resonance_value":0.0,
        "hf_lock_zero_crossing_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_side_sigma_multiplier":1.0, 
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "side_high_mag_um_per_pixel":PI_DUMMY
    }
    run_param_values = {
        "ImagFreq0":0.0
    }
    fun_kwargs = {
        "state_index":1
    }
    _get_atom_counts_test_helper("side_high_mag", analysis_functions.get_atom_count_side_li_hf, 
                        experiment_param_values = hf_atom_density_experiment_param_values, run_param_values = run_param_values, fun_kwargs = fun_kwargs)

def test_get_atom_count_top_A_abs():
    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_2_unitarity_res_freq_MHz":0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0.0,
        "hf_lock_rr_resonance_value":0.0,
        "hf_lock_zero_crossing_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_side_sigma_multiplier":1.0, 
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "top_um_per_pixel":SQRT_2_DUMMY
    }
    run_param_values = {
        "ImagFreq1":0.0
    }
    fun_kwargs = {
        "state_index":1
    }
    _get_atom_counts_test_helper("top_double", analysis_functions.get_atom_count_top_A_abs, 
                    experiment_param_values = hf_atom_density_experiment_param_values, run_param_values = run_param_values, 
                    fun_kwargs = fun_kwargs)
    
def test_get_atom_count_top_B_abs():
    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_2_unitarity_res_freq_MHz":0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0.0,
        "hf_lock_rr_resonance_value":0.0,
        "hf_lock_zero_crossing_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_side_sigma_multiplier":1.0, 
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "top_um_per_pixel":SQRT_2_DUMMY
    }
    run_param_values = {
        "ImagFreq2":0.0
    }
    fun_kwargs = {
        "state_index":3
    }
    _get_atom_counts_test_helper("top_double", analysis_functions.get_atom_count_top_B_abs, 
                    experiment_param_values = hf_atom_density_experiment_param_values, run_param_values = run_param_values, 
                    fun_kwargs = fun_kwargs)

def test_get_atom_counts_top_AB_abs():

    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_2_unitarity_res_freq_MHz":0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0.0,
        "hf_lock_rr_resonance_value":0.0,
        "hf_lock_zero_crossing_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_side_sigma_multiplier":1.0, 
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "top_um_per_pixel":SQRT_2_DUMMY
    }
    run_param_values = {
        "ImagFreq1":0.0,
        "ImagFreq2":0.0
    }
    fun_kwargs= {
        "state_index":1
    }

    def top_abs_A_split_off(my_measurement, my_run, **fun_kwargs):
        fun_kwargs["first_state_index"] = fun_kwargs.pop("state_index")
        return analysis_functions.get_atom_counts_top_AB_abs(my_measurement, my_run, **fun_kwargs)[0] 
    
    def top_abs_B_split_off(my_measurement, my_run, **fun_kwargs):
        fun_kwargs["second_state_index"] = fun_kwargs.pop("state_index")
        return analysis_functions.get_atom_counts_top_AB_abs(my_measurement, my_run, **fun_kwargs)[1] 
    
    _get_atom_counts_test_helper("top_double", top_abs_A_split_off, 
                    experiment_param_values = hf_atom_density_experiment_param_values, run_param_values = run_param_values, 
                    fun_kwargs = fun_kwargs)

    _get_atom_counts_test_helper("top_double", top_abs_B_split_off, 
                    experiment_param_values = hf_atom_density_experiment_param_values, run_param_values = run_param_values, 
                    fun_kwargs = fun_kwargs)
    
def test_get_atom_counts_top_polrot():
    experiment_param_values_polrot = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_2_unitarity_res_freq_MHz":10.0,
        "state_3_unitarity_res_freq_MHz":20.0,
        "hf_lock_unitarity_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_side_sigma_multiplier":1.0, 
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "top_um_per_pixel":SQRT_2_DUMMY,
        "polrot_phase_sign":1.0, 
        "top_camera_quantum_efficiency":0.7,
        "top_camera_counts_per_photoelectron":2.5,
        "top_camera_post_atom_photon_transmission":0.8,
        "top_camera_saturation_ramsey_fudge":1.1
    }
    run_param_values_polrot = {
        "ImagFreq1":3.0, 
        "ImagFreq2":23.0,
        "ImageTime":500.0
    }
    default_absorption_image = get_default_absorption_image()
    default_image_stack = generate_image_stack_from_absorption(default_absorption_image)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = default_image_stack,
                                                        run_param_values = run_param_values_polrot, experiment_param_values = experiment_param_values_polrot, 
                                                        ROI = DEFAULT_ABSORPTION_IMAGE_ROI, norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        #Already validated that get_atom_densities works; just validate that the counting is correct
        polrot_density_1, polrot_density_2 = analysis_functions.get_atom_densities_top_polrot(my_measurement, my_run, first_state_index = 1, 
                                                                                              second_state_index = 3, b_field_condition = "unitarity", 
                                                                                              use_saturation = False)
        um_per_pixel = experiment_param_values_polrot["top_um_per_pixel"]
        polrot_expected_counts_1 = np.sum(polrot_density_1) * np.square(um_per_pixel)
        polrot_expected_counts_2 = np.sum(polrot_density_2) * np.square(um_per_pixel)
        polrot_counts_1, polrot_counts_2 = analysis_functions.get_atom_counts_top_polrot(my_measurement, my_run, first_state_index = 1, 
                                                                                         second_state_index = 3, b_field_condition = "unitarity", 
                                                                                         use_saturation = False)
        assert np.isclose(polrot_counts_1, polrot_expected_counts_1)
        assert np.isclose(polrot_counts_2, polrot_expected_counts_2)
        #Then do it with saturation, just to check piping
        polrot_density_1_sat, polrot_density_2_sat = analysis_functions.get_atom_densities_top_polrot(my_measurement, my_run, first_state_index = 1, 
                                                                                              second_state_index = 3, b_field_condition = "unitarity", 
                                                                                              use_saturation = True)
        polrot_expected_counts_1_sat = np.sum(polrot_density_1_sat) * np.square(um_per_pixel)
        polrot_expected_counts_2_sat = np.sum(polrot_density_2_sat) * np.square(um_per_pixel)
        polrot_counts_1_sat, polrot_counts_2_sat = analysis_functions.get_atom_counts_top_polrot(my_measurement, my_run, first_state_index = 1, 
                                                                                         second_state_index = 3, b_field_condition = "unitarity", 
                                                                                         use_saturation = True)
        assert np.isclose(polrot_counts_1_sat, polrot_expected_counts_1_sat)
        assert np.isclose(polrot_counts_2_sat, polrot_expected_counts_2_sat)
    finally:
        shutil.rmtree(measurement_pathname)

def test_get_hybrid_trap_densities_along_harmonic_axis():
    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "top_um_per_pixel":SQRT_2_DUMMY, 
        "axicon_diameter_pix":100,
        "axicon_tilt_deg":5.0,
        "axicon_side_aspect_ratio":1.0, 
        "axicon_side_angle_deg":0.0,
        "hybrid_trap_typical_length_pix":DEFAULT_ABS_SQUARE_WIDTH,
        "axial_trap_frequency_hz":E_DUMMY
    }
    run_param_values = {
        "ImagFreq1":0.0, 
        "ImagFreq2":0.0
    }
    hybrid_sample_image = get_hybrid_sample_absorption_image()
    hybrid_sample_image_stack = generate_image_stack_from_absorption(hybrid_sample_image)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = hybrid_sample_image_stack, 
                                                        run_param_values = run_param_values, experiment_param_values = hf_atom_density_experiment_param_values, 
                                                        ROI = DEFAULT_ABSORPTION_IMAGE_ROI, norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        #Test and compare gettting vs. storing the densities 
        my_measurement.analyze_runs(analysis_functions.get_atom_densities_top_abs, ("densities_1", "densities_3"))
        hybrid_integrated_values_stored_densities = analysis_functions.get_hybrid_trap_densities_along_harmonic_axis(my_measurement, my_run, 
                                                                            imaging_mode = "abs", autocut = False,  
                                                                            first_stored_density_name = "densities_1", 
                                                                            second_stored_density_name = "densities_3")
        hybrid_integrated_values_unstored_densities = analysis_functions.get_hybrid_trap_densities_along_harmonic_axis(my_measurement, my_run, 
                                                                            imaging_mode = "abs", autocut = False)
        assert np.all(np.isclose(hybrid_integrated_values_stored_densities, hybrid_integrated_values_unstored_densities))

        hybrid_integrated_densities_uncut = hybrid_integrated_values_stored_densities 
        #Return type is a tuple containing positions and densities
        hybrid_integrated_positions, hybrid_integrated_density_uncut, _, _ = hybrid_integrated_densities_uncut
        cropped_hybrid_sample_image = get_hybrid_sample_absorption_image(crop_to_roi=True) 
        cropped_hybrid_sample_densities = -np.log(cropped_hybrid_sample_image) / li_6_res_cross_section
        um_per_pixel = my_measurement.experiment_parameters["top_um_per_pixel"]
        box_radius_um = um_per_pixel * my_measurement.experiment_parameters["axicon_diameter_pix"] / 2
        box_cross_section_um = np.pi * np.square(box_radius_um)
        #Correct for rotation, via adjustment to the cross section
        angle_adjusted_cross_section_um = box_cross_section_um / np.cos(np.deg2rad(my_measurement.experiment_parameters["axicon_tilt_deg"]))
        expected_hybrid_integrated_densities = np.sum(cropped_hybrid_sample_densities, axis = 1) * um_per_pixel / angle_adjusted_cross_section_um
        assert np.all(np.isclose(hybrid_integrated_density_uncut, expected_hybrid_integrated_densities, rtol = 1e-3, atol = 1e-4))
        #Also check that the positions are of the correct length and spacing
        assert len(hybrid_integrated_positions) == DEFAULT_ABSORPTION_IMAGE_ROI_SHAPE[0] 
        assert np.all(np.isclose(np.diff(hybrid_integrated_positions), um_per_pixel))
        #Lastly, test that autocrop works. Note: we are NOT trying to test the logic of autocrop, just the piping
        autocrop_start_index, autocrop_stop_index = science_functions.hybrid_trap_autocut(expected_hybrid_integrated_densities)
        expected_hybrid_integrated_densities_autocropped = expected_hybrid_integrated_densities[autocrop_start_index:autocrop_stop_index]
        hybrid_integrated_values_autocropped = analysis_functions.get_hybrid_trap_densities_along_harmonic_axis(my_measurement, my_run, 
                                                                            imaging_mode = "abs", autocut = True,  
                                                                            first_stored_density_name = "densities_1", 
                                                                            second_stored_density_name = "densities_3")
        _, hybrid_integrated_density_autocropped, _, _ = hybrid_integrated_values_autocropped 
        assert np.all(np.isclose(hybrid_integrated_density_autocropped, expected_hybrid_integrated_densities_autocropped, rtol = 1e-3, atol = 1e-4))
        #Also check that the potential return option works
        hybrid_values_potential_returned =analysis_functions.get_hybrid_trap_densities_along_harmonic_axis(my_measurement, my_run, 
                                                                            imaging_mode = "abs", autocut = False, return_potentials = True, 
                                                                            first_stored_density_name = "densities_1", 
                                                                            second_stored_density_name = "densities_3")
        _, hybrid_integrated_potentials, *_ = hybrid_values_potential_returned
        expected_hybrid_integrated_potentials = science_functions.get_li_energy_hz_in_1D_trap(hybrid_integrated_positions * 1e-6,
                                                        hf_atom_density_experiment_param_values["axial_trap_frequency_hz"])
        assert np.all(np.isclose(hybrid_integrated_potentials, expected_hybrid_integrated_potentials))
    finally:
        shutil.rmtree(measurement_pathname)


def test_hybrid_trap_density_helper():
    EXPECTED_X_CENTER = 228 
    EXPECTED_Y_CENTER = 398
    SAMPLE_UM_PER_PIXEL = 0.6
    SAMPLE_AXICON_DIAMETER_PIX = 189
    SAMPLE_AXICON_LENGTH_PIX = 250
    SAMPLE_TILT_DEG = 6.3
    SAMPLE_SIDE_TILT_DEG = 10
    SAMPLE_SIDE_ASPECT_RATIO = 1.4
    hybrid_trap_experiment_parameters = {
        "top_um_per_pixel":SAMPLE_UM_PER_PIXEL, 
        "axicon_diameter_pix":SAMPLE_AXICON_DIAMETER_PIX,
        "axicon_tilt_deg":SAMPLE_TILT_DEG,
        "axicon_side_aspect_ratio":SAMPLE_SIDE_ASPECT_RATIO, 
        "axicon_side_angle_deg": SAMPLE_SIDE_TILT_DEG,
        "hybrid_trap_typical_length_pix":SAMPLE_AXICON_LENGTH_PIX,
    }
    try:
        measurement_pathname, my_measurement, _ = create_measurement("top_double", experiment_param_values = hybrid_trap_experiment_parameters)
        sample_hybrid_trap_data = np.load('resources/Sample_Box_Exp.npy')
        hybrid_trap_harmonic_positions, hybrid_trap_harmonic_data = analysis_functions._hybrid_trap_density_helper(my_measurement, sample_hybrid_trap_data)
        filtered_hybrid_trap_harmonic_data = scipy.signal.savgol_filter(hybrid_trap_harmonic_data, 15, 2)
        max_index = np.argmax(hybrid_trap_harmonic_data) 
        max_value = hybrid_trap_harmonic_data[max_index]
        CENTER_SNIPPET_HALF_WIDTH = 10
        center_snippet = sample_hybrid_trap_data[EXPECTED_Y_CENTER - CENTER_SNIPPET_HALF_WIDTH:EXPECTED_Y_CENTER+CENTER_SNIPPET_HALF_WIDTH, 
                                                EXPECTED_X_CENTER - CENTER_SNIPPET_HALF_WIDTH: EXPECTED_X_CENTER + CENTER_SNIPPET_HALF_WIDTH]
        center_snippet_average_2d_density = np.sum(center_snippet) / center_snippet.size
        center_snippet_average_3d_density = center_snippet_average_2d_density / (SAMPLE_UM_PER_PIXEL * SAMPLE_AXICON_DIAMETER_PIX)
        max_index = np.argmax(filtered_hybrid_trap_harmonic_data)
        max_value = filtered_hybrid_trap_harmonic_data[max_index] 
        max_position = hybrid_trap_harmonic_positions[max_index] 
        assert(np.abs(max_position) < 10) 
        assert(np.abs((center_snippet_average_3d_density - max_value) / center_snippet_average_3d_density < 1e-1))
    finally:
        shutil.rmtree(measurement_pathname)


def test_rotate_and_crop_hybrid_image():
    X_SIZE = 300 
    Y_SIZE = 300 
    X_CENTER = 110 
    Y_CENTER = 203 
    GAUSSIAN_X_WIDTH = 50
    GAUSSIAN_Y_WIDTH = 20
    center = (X_CENTER, Y_CENTER)
    y_indices, x_indices = np.mgrid[0:Y_SIZE, 0:X_SIZE]
    gaussian_data = data_fitting_functions.two_dimensional_gaussian(x_indices, y_indices, 1.0, X_CENTER, Y_CENTER, GAUSSIAN_X_WIDTH, GAUSSIAN_Y_WIDTH, 0)
    X_CROP_WIDTH = 100 
    Y_CROP_WIDTH = 150
    cropped_rotated_image, cropped_rotated_center = analysis_functions._rotate_and_crop_hybrid_image(gaussian_data, center, 90, 
                                                                                x_crop_width = X_CROP_WIDTH, y_crop_width = Y_CROP_WIDTH)
    rotated_x_center, rotated_y_center = cropped_rotated_center 
    assert (np.abs(rotated_x_center - X_CROP_WIDTH / 2.0 < 1e-3))
    assert (np.abs(rotated_y_center - Y_CROP_WIDTH / 2.0 < 1e-3))
    EXPECTED_COUNTS_SUM = 5375.8749
    counts_sum = np.sum(cropped_rotated_image)
    assert (np.abs(counts_sum - EXPECTED_COUNTS_SUM) < 1e-3)


def test_get_hybrid_trap_average_energy():
    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "top_um_per_pixel":SQRT_2_DUMMY, 
        "axicon_diameter_pix":100,
        "axicon_tilt_deg":5.0,
        "axicon_side_aspect_ratio":1.0, 
        "axicon_side_angle_deg":0.0,
        "hybrid_trap_typical_length_pix":DEFAULT_ABS_SQUARE_WIDTH,
        "axial_trap_frequency_hz":E_DUMMY
    }
    run_param_values = {
        "ImagFreq1":0.0,
        "ImagFreq2":0.0
    }
    hybrid_sample_image = get_hybrid_sample_absorption_image()
    hybrid_sample_image_stack = generate_image_stack_from_absorption(hybrid_sample_image)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = hybrid_sample_image_stack, 
                                                        run_param_values = run_param_values, experiment_param_values = hf_atom_density_experiment_param_values, 
                                                        ROI = DEFAULT_ABSORPTION_IMAGE_ROI, norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        
        #Test and compare gettting vs. storing the densities 
        my_measurement.analyze_runs(analysis_functions.get_atom_densities_top_abs, ("densities_1", "densities_3"))
        hybrid_energy_stored_densities = analysis_functions.get_hybrid_trap_average_energy(my_measurement, my_run, 
                                                                            autocut = True,  
                                                                            first_stored_density_name = "densities_1", 
                                                                            second_stored_density_name = "densities_3")
        hybrid_energy_unstored_densities = analysis_functions.get_hybrid_trap_average_energy(my_measurement, my_run, 
                                                                            imaging_mode = "abs", autocut = True)
        assert np.isclose(hybrid_energy_stored_densities, hybrid_energy_unstored_densities)
        average_energy_autocut = hybrid_energy_stored_densities

        # #Return type is a tuple containing positions and densities
        cropped_hybrid_sample_image = get_hybrid_sample_absorption_image(crop_to_roi=True) 
        cropped_hybrid_sample_densities = -np.log(cropped_hybrid_sample_image) / li_6_res_cross_section
        um_per_pixel = my_measurement.experiment_parameters["top_um_per_pixel"]
        box_radius_um = um_per_pixel * my_measurement.experiment_parameters["axicon_diameter_pix"] / 2
        box_cross_section_um = np.pi * np.square(box_radius_um)
        angle_adjusted_cross_section_um = box_cross_section_um / np.cos(np.deg2rad(my_measurement.experiment_parameters["axicon_tilt_deg"]))
        expected_hybrid_integrated_densities = np.sum(cropped_hybrid_sample_densities, axis = 1) * um_per_pixel / angle_adjusted_cross_section_um 
        expected_densities_length = len(expected_hybrid_integrated_densities)
        expected_hybrid_integrated_positions = (np.arange(expected_densities_length) - (expected_densities_length - 1) // 2) * um_per_pixel
        expected_average_energy_uncropped = science_functions.get_hybrid_trap_average_energy(
            expected_hybrid_integrated_positions, expected_hybrid_integrated_densities, 
            angle_adjusted_cross_section_um, my_measurement.experiment_parameters["axial_trap_frequency_hz"])
        average_energy_uncropped = analysis_functions.get_hybrid_trap_average_energy(my_measurement, my_run, 
                                                                                     autocut = False, 
                                                                                     first_stored_density_name = "densities_1", 
                                                                                     second_stored_density_name = "densities_3")
        assert np.isclose(average_energy_uncropped, expected_average_energy_uncropped)
        #Also compare the energy after autocutting 
        autocut_start_index, autocut_stop_index = science_functions.hybrid_trap_autocut(expected_hybrid_integrated_densities) 
        expected_hybrid_integrated_densities_autocut = expected_hybrid_integrated_densities[autocut_start_index:autocut_stop_index]
        expected_hybrid_integrated_positions_autocut = expected_hybrid_integrated_positions[autocut_start_index:autocut_stop_index] 
        expected_average_energy_autocut = science_functions.get_hybrid_trap_average_energy(
            expected_hybrid_integrated_positions_autocut, expected_hybrid_integrated_densities_autocut, 
            angle_adjusted_cross_section_um, my_measurement.experiment_parameters["axial_trap_frequency_hz"])
        assert np.isclose(expected_average_energy_autocut, average_energy_autocut)
    finally:
        shutil.rmtree(measurement_pathname)

def test_get_hybrid_trap_compressibilities():
    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "top_um_per_pixel":SQRT_2_DUMMY, 
        "axicon_diameter_pix":100,
        "axicon_tilt_deg":0.0,
        "axicon_side_aspect_ratio":1.0, 
        "axicon_side_angle_deg":0.0,
        "axial_trap_frequency_hz":E_DUMMY,
        "hybrid_trap_typical_length_pix":DEFAULT_ABS_SQUARE_WIDTH
    }
    run_param_values = {
        "ImagFreq1":0.0, 
        "ImagFreq2":0.0
    }
    hybrid_sample_image = get_hybrid_sample_absorption_image()
    hybrid_sample_image_stack = generate_image_stack_from_absorption(hybrid_sample_image)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = hybrid_sample_image_stack, 
                                                        run_param_values = run_param_values, experiment_param_values = hf_atom_density_experiment_param_values, 
                                                        ROI = DEFAULT_ABSORPTION_IMAGE_ROI, norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        my_measurement.analyze_runs(analysis_functions.get_atom_densities_top_abs, ("densities_1", "densities_3"))
        positions_1, potentials_1, densities_1, positions_2, potentials_2, densities_2 = analysis_functions.get_hybrid_trap_densities_along_harmonic_axis(
                my_measurement, my_run, imaging_mode = "abs", autocut = False, return_potentials = True,
                first_stored_density_name = "densities_1", second_stored_density_name = "densities_3"
            )
        
        TEST_WINDOW_SIZE = 15
        expected_index_breakpoints_1 = np.arange(0, len(potentials_1), TEST_WINDOW_SIZE)
        expected_position_midpoints_1 = (positions_1[expected_index_breakpoints_1][:-1] + positions_1[expected_index_breakpoints_1 - 1][1:]) / 2.0 
        expected_potential_midpoints_1 = (potentials_1[expected_index_breakpoints_1][:-1] + potentials_1[expected_index_breakpoints_1 - 1][1:]) / 2.0
        expected_compressibilities_1, expected_errors_1 = science_functions.get_hybrid_trap_compressibilities_window_fit(potentials_1, densities_1, 
                                                                            expected_index_breakpoints_1, return_errors = True)
        
        expected_index_breakpoints_2 = np.arange(0, len(potentials_2), TEST_WINDOW_SIZE)
        expected_position_midpoints_2 = (positions_2[expected_index_breakpoints_2][:-1] + positions_2[expected_index_breakpoints_2 - 1][1:]) / 2.0 
        expected_potential_midpoints_2 = (potentials_2[expected_index_breakpoints_2][:-1] + potentials_2[expected_index_breakpoints_2 - 1][1:]) / 2.0
        expected_compressibilities_2, expected_errors_2 = science_functions.get_hybrid_trap_compressibilities_window_fit(potentials_2, densities_2, 
                                                                            expected_index_breakpoints_2, return_errors = True)

        (extracted_positions_1, extracted_potentials_1, extracted_compressibilities_1, extracted_errors_1,
            extracted_positions_2, extracted_potentials_2, 
            extracted_compressibilities_2, extracted_errors_2) = analysis_functions.get_hybrid_trap_compressibilities(my_measurement, my_run, 
                                                    first_stored_density_name = "densities_1", second_stored_density_name = "densities_3", 
                                                    return_errors = True, return_positions = True, return_potentials = True, 
                                                    window_size = TEST_WINDOW_SIZE)
        assert np.all(np.isclose(extracted_compressibilities_1, expected_compressibilities_1))
        assert np.all(np.isclose(extracted_errors_1, expected_errors_1))
        assert np.all(np.isclose(expected_position_midpoints_1, extracted_positions_1))
        assert np.all(np.isclose(expected_potential_midpoints_1, extracted_potentials_1))

        assert np.all(np.isclose(extracted_compressibilities_2, expected_compressibilities_2))
        assert np.all(np.isclose(extracted_errors_2, expected_errors_2))
        assert np.all(np.isclose(expected_position_midpoints_2, extracted_positions_2))
        assert np.all(np.isclose(expected_potential_midpoints_2, extracted_potentials_2))

    finally:
        shutil.rmtree(measurement_pathname)


def test_get_axial_squish_densities_along_harmonic_axis():
    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "top_um_per_pixel":SQRT_2_DUMMY, 
        "axicon_diameter_pix":100,
        "axicon_tilt_deg":0.0,
        "axicon_side_aspect_ratio":1.0, 
        "axicon_side_angle_deg":0.0,
        "axial_trap_frequency_hz":E_DUMMY,
        "hybrid_trap_typical_length_pix":DEFAULT_ABS_SQUARE_WIDTH,
        "hybrid_trap_center_pix_polrot":255,
        "axial_gradient_Hz_per_um_V":SQRT_5_DUMMY
    }
    run_param_values = {
        "ImagFreq1":0.0, 
        "ImagFreq2":0.0,
        "Axial_Squish_Imaging_Grad_V":SQRT_7_DUMMY
    }
    hybrid_sample_image = get_hybrid_sample_absorption_image()
    hybrid_sample_image_stack = generate_image_stack_from_absorption(hybrid_sample_image)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = hybrid_sample_image_stack, 
                                                        run_param_values = run_param_values, experiment_param_values = hf_atom_density_experiment_param_values, 
                                                        ROI = DEFAULT_ABSORPTION_IMAGE_ROI, norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        my_measurement.analyze_runs(analysis_functions.get_atom_densities_top_abs, ("densities_1", "densities_3"))
        densities_1, densities_2 = analysis_functions.get_hybrid_trap_densities_along_harmonic_axis(
                my_measurement, my_run, imaging_mode = "abs", autocut = False, return_positions = False,
                first_stored_density_name = "densities_1", second_stored_density_name = "densities_3"
            )
        
        #First test without autocutting
        expected_uncut_indices = np.arange(len(densities_1))
        absolute_harmonic_center = my_measurement.experiment_parameters["hybrid_trap_center_pix_polrot"]
        _, ymin, *_ = my_measurement.measurement_parameters["ROI"]
        relative_harmonic_center = absolute_harmonic_center - ymin
        expected_uncut_referenced_index_positions = expected_uncut_indices - relative_harmonic_center
        expected_uncut_positions = expected_uncut_referenced_index_positions * my_measurement.experiment_parameters["top_um_per_pixel"]
        gradient_potential = (expected_uncut_positions *
                 my_measurement.experiment_parameters["axial_gradient_Hz_per_um_V"] * my_run.parameters["Axial_Squish_Imaging_Grad_V"])
        trap_freq = my_measurement.experiment_parameters["axial_trap_frequency_hz"]
        harmonic_potential = science_functions.get_li_energy_hz_in_1D_trap(expected_uncut_positions * 1e-6, trap_freq)
        expected_uncut_potential = gradient_potential + harmonic_potential

        (uncut_positions_1, uncut_potentials_1, uncut_densities_1,
         uncut_positions_2, uncut_potentials_2, uncut_densities_2) = analysis_functions.get_axial_squish_densities_along_harmonic_axis(
                                                        my_measurement, my_run, autocut = False, return_positions = True, return_potentials = True, 
                                                        first_stored_density_name = "densities_1", second_stored_density_name = "densities_3")
        assert np.all(np.isclose(uncut_positions_1, expected_uncut_positions))
        assert np.all(np.isclose(uncut_potentials_1, expected_uncut_potential))
        assert np.all(np.isclose(uncut_densities_1, densities_1))
        assert np.all(np.isclose(uncut_positions_2, expected_uncut_positions))
        assert np.all(np.isclose(uncut_potentials_2, expected_uncut_potential))
        assert np.all(np.isclose(uncut_densities_2, densities_2))

        #Now test with autocutting, just the densities
        HARDCODED_LOWER_CUT_BUFFER = 5
        expected_low_cut_index = np.argmax(densities_1) + HARDCODED_LOWER_CUT_BUFFER
        _, expected_high_cut_index_1 = science_functions.hybrid_trap_autocut(densities_1)
        _, expected_high_cut_index_2 = science_functions.hybrid_trap_autocut(densities_2)
        expected_autocut_densities_1 = densities_1[expected_low_cut_index:expected_high_cut_index_1]
        expected_autocut_densities_2 = densities_2[expected_low_cut_index:expected_high_cut_index_2] 
        expected_autocut_potentials_1 = expected_uncut_potential[expected_low_cut_index:expected_high_cut_index_1]
        expected_autocut_potentials_2 = expected_uncut_potential[expected_low_cut_index:expected_high_cut_index_2]
        expected_autocut_positions_1 = expected_uncut_positions[expected_low_cut_index:expected_high_cut_index_1] 
        expected_autocut_positions_2 = expected_uncut_positions[expected_low_cut_index:expected_high_cut_index_2]
        (cut_positions_1, cut_potentials_1, cut_densities_1,
         cut_positions_2, cut_potentials_2, cut_densities_2) = analysis_functions.get_axial_squish_densities_along_harmonic_axis(
                                                        my_measurement, my_run, autocut = True, return_positions = True, return_potentials = True, 
                                                        first_stored_density_name = "densities_1", second_stored_density_name = "densities_3")
        assert np.all(np.isclose(cut_positions_1, expected_autocut_positions_1))
        assert np.all(np.isclose(cut_potentials_1, expected_autocut_potentials_1))
        assert np.all(np.isclose(cut_densities_1, expected_autocut_densities_1))
        assert np.all(np.isclose(cut_positions_2, expected_autocut_positions_2))
        assert np.all(np.isclose(cut_potentials_2, expected_autocut_potentials_2))
        assert np.all(np.isclose(cut_densities_2, expected_autocut_densities_2))
    finally:
        shutil.rmtree(measurement_pathname)


def test_get_axial_squish_absolute_pressures():
    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "top_um_per_pixel":SQRT_2_DUMMY, 
        "axicon_diameter_pix":100,
        "axicon_tilt_deg":0.0,
        "axicon_side_aspect_ratio":1.0, 
        "axicon_side_angle_deg":0.0,
        "axial_trap_frequency_hz":E_DUMMY,
        "hybrid_trap_typical_length_pix":DEFAULT_ABS_SQUARE_WIDTH,
        "hybrid_trap_center_pix_polrot":255,
        "axial_gradient_Hz_per_um_V":SQRT_5_DUMMY
    }
    run_param_values = {
        "ImagFreq1":0.0, 
        "ImagFreq2":0.0,
        "Axial_Squish_Imaging_Grad_V":SQRT_7_DUMMY
    }
    hybrid_sample_image = get_hybrid_sample_absorption_image()
    hybrid_sample_image_stack = generate_image_stack_from_absorption(hybrid_sample_image)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = hybrid_sample_image_stack, 
                                                        run_param_values = run_param_values, experiment_param_values = hf_atom_density_experiment_param_values, 
                                                        ROI = DEFAULT_ABSORPTION_IMAGE_ROI, norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        my_measurement.analyze_runs(analysis_functions.get_atom_densities_top_abs, ("densities_1", "densities_3"))
        my_measurement.analyze_runs(analysis_functions.get_axial_squish_densities_along_harmonic_axis, 
                        ("positions_1", "potentials_1", "ax_densities_1", "positions_3", "potentials_3", "ax_densities_3"), fun_kwargs = {
                            "return_positions":True,
                            "autocut":True,
                            "first_stored_density_name":"densities_1", 
                            "second_stored_density_name":"densities_3"
                        })
        positions_1, potentials_1, densities_1, positions_3, potentials_3, densities_3 = my_measurement.get_analysis_value_from_runs(
            ("positions_1", "potentials_1", "ax_densities_1", "positions_3", "potentials_3", "ax_densities_3")
        )

        potentials_1 = potentials_1[0] 
        potentials_3 = potentials_3[0]

        expected_pressures_1 = science_functions.get_absolute_pressures(potentials_1, densities_1)
        expected_pressures_3 = science_functions.get_absolute_pressures(potentials_3, densities_3)
        (extracted_positions_1, extracted_potentials_1, extracted_densities_1, extracted_pressures_1, 
         extracted_positions_3, extracted_potentials_3,
           extracted_densities_3, extracted_pressures_3) = analysis_functions.get_axial_squish_absolute_pressures(
               my_measurement, my_run, return_positions = True, return_potentials = True, return_densities = True,
               first_stored_density_name = "densities_1", second_stored_density_name = "densities_3"
           )
        assert np.all(np.isclose(positions_1, extracted_positions_1))
        assert np.all(np.isclose(positions_3, extracted_positions_3))
        assert np.all(np.isclose(potentials_1, extracted_potentials_1))
        assert np.all(np.isclose(potentials_3, extracted_potentials_3))
        assert np.all(np.isclose(densities_1, extracted_densities_1))
        assert np.all(np.isclose(densities_3, extracted_densities_3))
        assert np.all(np.isclose(expected_pressures_1, extracted_pressures_1))
        assert np.all(np.isclose(expected_pressures_3, extracted_pressures_3))
    finally:
        shutil.rmtree(measurement_pathname)

def test_get_axial_squish_normalized_pressures():
    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "top_um_per_pixel":SQRT_2_DUMMY, 
        "axicon_diameter_pix":100,
        "axicon_tilt_deg":0.0,
        "axicon_side_aspect_ratio":1.0, 
        "axicon_side_angle_deg":0.0,
        "axial_trap_frequency_hz":E_DUMMY,
        "hybrid_trap_typical_length_pix":DEFAULT_ABS_SQUARE_WIDTH,
        "hybrid_trap_center_pix_polrot":255,
        "axial_gradient_Hz_per_um_V":SQRT_5_DUMMY
    }
    run_param_values = {
        "ImagFreq1":0.0, 
        "ImagFreq2":0.0,
        "Axial_Squish_Imaging_Grad_V":SQRT_7_DUMMY
    }
    hybrid_sample_image = get_hybrid_sample_absorption_image_norm_pressure()
    hybrid_sample_image_stack = generate_image_stack_from_absorption(hybrid_sample_image)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = hybrid_sample_image_stack, 
                                                        run_param_values = run_param_values, experiment_param_values = hf_atom_density_experiment_param_values, 
                                                        ROI = HYBRID_AVOID_ZERO_ROI, norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        my_measurement.analyze_runs(analysis_functions.get_atom_densities_top_abs, ("densities_1", "densities_3"))
        my_measurement.analyze_runs(analysis_functions.get_axial_squish_densities_along_harmonic_axis, 
                        ("ax_densities_1", "ax_densities_3"), fun_kwargs = {
                            "return_positions":False,
                            "return_potentials":False,
                            "autocut":True,
                            "first_stored_density_name":"densities_1", 
                            "second_stored_density_name":"densities_3"
                        })
        my_measurement.analyze_runs(analysis_functions.get_axial_squish_absolute_pressures, 
                                    ("potentials_1", "abs_pressure_1", "potentials_3", "abs_pressure_3"), fun_kwargs = {
                                        "autocut":True,
                                        "first_stored_density_name":"densities_1",
                                        "second_stored_density_name":"densities_3"
                                    })
        expected_potentials_1, abs_pressure_1, expected_potentials_3, abs_pressure_3 = my_measurement.get_analysis_value_from_runs(
            ("potentials_1", "abs_pressure_1", "potentials_3", "abs_pressure_3")
        )

        expected_potentials_1 = expected_potentials_1.flatten()
        expected_potentials_3 = expected_potentials_3.flatten()
        abs_pressure_1 = abs_pressure_1.flatten()
        abs_pressure_3 = abs_pressure_3.flatten()

        axial_densities_1, axial_densities_3 = my_measurement.get_analysis_value_from_runs(
            ("ax_densities_1", "ax_densities_3")
        )

        axial_densities_1 = axial_densities_1.flatten()
        axial_densities_3 = axial_densities_3.flatten()

        pressure_denominator_1 = eos_functions.fermi_pressure_Hz_um_from_density_um(axial_densities_1)
        pressure_denominator_3 = eos_functions.fermi_pressure_Hz_um_from_density_um(axial_densities_3)
        expected_norm_pressure_1 = abs_pressure_1 / pressure_denominator_1
        expected_norm_pressure_3 = abs_pressure_3 / pressure_denominator_3
        #First test with the additional normalized pressure cut turned off
        potential_1, norm_pressure_1, potential_3, norm_pressure_3 = analysis_functions.get_axial_squish_normalized_pressures(
            my_measurement, my_run, return_positions = False, return_potentials = True, autocut = True,
            first_stored_density_name = "densities_1", second_stored_density_name = "densities_3",
            normalized_pressure_cut = False
        )

        potential_1 = potential_1.flatten() 
        potential_3 = potential_3.flatten() 
        norm_pressure_1 = norm_pressure_1.flatten() 
        norm_pressure_3 = norm_pressure_3.flatten()

        assert np.allclose(potential_1, expected_potentials_1)
        assert np.allclose(potential_3, expected_potentials_3)
        assert np.allclose(expected_norm_pressure_1, norm_pressure_1)
        assert np.allclose(expected_norm_pressure_3, norm_pressure_3)
        #Now try autocutting... 

        norm_pressure_autocut_point = 0.1 

        potential_1_cut, norm_pressure_1_cut, potential_3_cut, norm_pressure_3_cut = analysis_functions.get_axial_squish_normalized_pressures(
            my_measurement, my_run, return_positions = False, return_potentials = True, autocut = True, 
            first_stored_density_name = "densities_1", second_stored_density_name = "densities_3",
            normalized_pressure_cut = True, normalized_pressure_relative_cut_point = norm_pressure_autocut_point)
        
        potential_1_cut = potential_1_cut.flatten()
        potential_3_cut = potential_3_cut.flatten()
        norm_pressure_1_cut = norm_pressure_1_cut.flatten()
        norm_pressure_3_cut = norm_pressure_3_cut.flatten()

        included_indices_1 = axial_densities_1 > (np.max(axial_densities_1) * norm_pressure_autocut_point)
        included_indices_3 = axial_densities_3 > (np.max(axial_densities_3) * norm_pressure_autocut_point)

        expected_potentials_1_autocut = expected_potentials_1[included_indices_1] 
        expected_potentials_3_autocut = expected_potentials_3[included_indices_3]
        expected_norm_pressure_1_autocut = expected_norm_pressure_1[included_indices_1] 
        expected_norm_pressure_3_autocut = expected_norm_pressure_3[included_indices_3]

        assert np.allclose(potential_1_cut, expected_potentials_1_autocut)
        assert np.allclose(potential_3_cut, expected_potentials_3_autocut)
        assert np.allclose(norm_pressure_1_cut, expected_norm_pressure_1_autocut)
        assert np.allclose(norm_pressure_3_cut, expected_norm_pressure_3_autocut)

    finally:
        shutil.rmtree(measurement_pathname)


def test_get_axial_squish_compressibilities():
    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "top_um_per_pixel":SQRT_2_DUMMY, 
        "axicon_diameter_pix":100,
        "axicon_tilt_deg":0.0,
        "axicon_side_aspect_ratio":1.0, 
        "axicon_side_angle_deg":0.0,
        "axial_trap_frequency_hz":E_DUMMY,
        "hybrid_trap_typical_length_pix":DEFAULT_ABS_SQUARE_WIDTH,
        "hybrid_trap_center_pix_polrot":255,
        "axial_gradient_Hz_per_um_V":SQRT_5_DUMMY
    }
    run_param_values = {
        "ImagFreq1":0.0, 
        "ImagFreq2":0.0,
        "Axial_Squish_Imaging_Grad_V":SQRT_7_DUMMY
    }
    hybrid_sample_image = get_hybrid_sample_absorption_image_norm_pressure()
    hybrid_sample_image_stack = generate_image_stack_from_absorption(hybrid_sample_image)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = hybrid_sample_image_stack, 
                                                        run_param_values = run_param_values, experiment_param_values = hf_atom_density_experiment_param_values, 
                                                        ROI = HYBRID_AVOID_ZERO_ROI, norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        my_measurement.analyze_runs(analysis_functions.get_atom_densities_top_abs, ("densities_1", "densities_3"))
        my_measurement.analyze_runs(analysis_functions.get_axial_squish_densities_along_harmonic_axis, 
                        ("potentials_1", "ax_densities_1", "potentials_3", "ax_densities_3"), fun_kwargs = {
                            "return_positions":False,
                            "return_potentials":True,
                            "autocut":False,
                            "first_stored_density_name":"densities_1", 
                            "second_stored_density_name":"densities_3"
                        })
        potentials_1, potentials_3, ax_densities_1, ax_densities_3 = my_measurement.get_analysis_value_from_runs(
            ("potentials_1", "potentials_3", "ax_densities_1", "ax_densities_3")
        )

        potentials_1 = potentials_1[0] 
        ax_densities_1 = ax_densities_1[0] 
        potentials_3 = potentials_3[0] 
        ax_densities_3 = ax_densities_3[0]

        COMPRESSIBILITY_WINDOW_SIZE = 19
        compressibility_index_breakpoints = np.arange(0, len(potentials_1), COMPRESSIBILITY_WINDOW_SIZE)
        expected_compressibilities_1, expected_errors_1 = science_functions.get_hybrid_trap_compressibilities_window_fit(
            potentials_1, ax_densities_1, compressibility_index_breakpoints, return_errors = True)
        expected_compressibilities_3, expected_errors_3 = science_functions.get_hybrid_trap_compressibilities_window_fit(
            potentials_3, ax_densities_3, compressibility_index_breakpoints, return_errors = True)
        

        expected_potentials_1 = 0.5 * (potentials_1[compressibility_index_breakpoints][:-1] + potentials_1[compressibility_index_breakpoints - 1][1:])
        expected_potentials_3 = 0.5 * (potentials_3[compressibility_index_breakpoints][:-1] + potentials_3[compressibility_index_breakpoints - 1][1:])

        (extracted_potentials_1, extracted_compressibilities_1, extracted_errors_1, 
        extracted_potentials_3, extracted_compressibilities_3, extracted_errors_3) = analysis_functions.get_axial_squish_compressibilities(
            my_measurement, my_run, first_stored_density_name = "densities_1", second_stored_density_name = "densities_3", 
            return_errors = True, return_potentials = True, window_size = COMPRESSIBILITY_WINDOW_SIZE
        )

        assert np.allclose(extracted_potentials_1, expected_potentials_1)
        assert np.allclose(extracted_potentials_3, expected_potentials_3) 
        assert np.allclose(extracted_compressibilities_1, expected_compressibilities_1)
        assert np.allclose(extracted_compressibilities_3, expected_compressibilities_3)
        assert np.allclose(extracted_errors_1, expected_errors_1)
        assert np.allclose(extracted_errors_3, expected_errors_3)

    finally:
        shutil.rmtree(measurement_pathname)


def test_get_axial_squish_compressibilities_vs_pressure():
    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "top_um_per_pixel":SQRT_2_DUMMY, 
        "axicon_diameter_pix":100,
        "axicon_tilt_deg":0.0,
        "axicon_side_aspect_ratio":1.0, 
        "axicon_side_angle_deg":0.0,
        "axial_trap_frequency_hz":E_DUMMY,
        "hybrid_trap_typical_length_pix":DEFAULT_ABS_SQUARE_WIDTH,
        "hybrid_trap_center_pix_polrot":255,
        "axial_gradient_Hz_per_um_V":SQRT_5_DUMMY
    }
    run_param_values = {
        "ImagFreq1":0.0, 
        "ImagFreq2":0.0,
        "Axial_Squish_Imaging_Grad_V":SQRT_7_DUMMY
    }
    hybrid_sample_image = get_hybrid_sample_absorption_image_norm_pressure()
    hybrid_sample_image_stack = generate_image_stack_from_absorption(hybrid_sample_image)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = hybrid_sample_image_stack, 
                                                        run_param_values = run_param_values, experiment_param_values = hf_atom_density_experiment_param_values, 
                                                        ROI = HYBRID_AVOID_ZERO_ROI, norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        my_measurement.analyze_runs(analysis_functions.get_atom_densities_top_abs, ("densities_1", "densities_3"))
        for autocut_setting in [True, False]:
            my_measurement.analyze_runs(analysis_functions.get_axial_squish_densities_along_harmonic_axis, 
                            ("potentials_1", "ax_densities_1", "potentials_3", "ax_densities_3"), fun_kwargs = {
                                "return_positions":False,
                                "return_potentials":True,
                                "autocut":False,
                                "first_stored_density_name":"densities_1", 
                                "second_stored_density_name":"densities_3"
                            })    
            my_measurement.analyze_runs(analysis_functions.get_axial_squish_normalized_pressures, ("norm_pressures_1", "norm_pressures_3"), 
                                fun_kwargs = {
                                    "first_stored_density_name":"densities_1", 
                                    "second_stored_density_name":"densities_3", 
                                    "autocut":autocut_setting,
                                    "normalized_pressure_cut":autocut_setting, 
                                    "return_positions":False, 
                                    "return_potentials":False})
            COMPRESSIBILITES_WINDOW_SIZE = 21 
            my_measurement.analyze_runs(analysis_functions.get_axial_squish_compressibilities, ("compressibilities_1", "compressibilities_3"), 
                                        fun_kwargs = {
                                            "first_stored_density_name":"densities_1", 
                                            "second_stored_density_name":"densities_3", 
                                            "autocut":autocut_setting, 
                                            "window_size":COMPRESSIBILITES_WINDOW_SIZE, 
                                            "return_potentials":False})
            
            ax_densities_1, ax_densities_3 = my_measurement.get_analysis_value_from_runs(("ax_densities_1", "ax_densities_3"))
            ax_densities_1 = ax_densities_1[0] 
            ax_densities_3 = ax_densities_3[0]
            imbalances = (ax_densities_3 - ax_densities_1) / (ax_densities_3 + ax_densities_1)

            norm_pressures_1, norm_pressures_3 = my_measurement.get_analysis_value_from_runs(("norm_pressures_1", "norm_pressures_3"))
            norm_pressures_1 = norm_pressures_1[0] 
            norm_pressures_3 = norm_pressures_3[0]

            expected_breakpoint_indices = np.arange(0, len(norm_pressures_1), COMPRESSIBILITES_WINDOW_SIZE)
            expected_midpoint_indices = np.round(0.5 * (expected_breakpoint_indices[1:] + expected_breakpoint_indices[:-1])).astype(int)
            expected_norm_pressures_1 = norm_pressures_1[expected_midpoint_indices] 
            expected_norm_pressures_3 = norm_pressures_3[expected_midpoint_indices]

            expected_imbalances = imbalances[expected_midpoint_indices] 

            expected_compressibilities_1, expected_compressibilities_3 = my_measurement.get_analysis_value_from_runs(
                ("compressibilities_1", "compressibilities_3"))
            
            expected_compressibilities_1 = expected_compressibilities_1[0] 
            expected_compressibilities_3 = expected_compressibilities_3[0] 

            (extracted_norm_pressures_1, extracted_imbalances_1, extracted_compressibilities_1,
            extracted_norm_pressures_3, extracted_imbalances_3, extracted_compressibilities_3) = analysis_functions.get_axial_squish_compressibilities_vs_pressure(
                my_measurement, my_run, 
                first_stored_density_name = "densities_1", 
                second_stored_density_name = "densities_3", 
                autocut = autocut_setting, normalized_pressure_cut = autocut_setting, 
                return_imbalances = True)
            
            #High rtol because actual method involves smoothing and interpolation - just need to check it's close
            assert np.allclose(expected_imbalances, extracted_imbalances_3)
            assert np.allclose(expected_imbalances, -1.0 * extracted_imbalances_1) 
            assert np.allclose(extracted_norm_pressures_1, expected_norm_pressures_1, rtol = 1e-1) 
            assert np.allclose(extracted_norm_pressures_3, expected_norm_pressures_3, rtol = 1e-1) 
            assert np.allclose(expected_compressibilities_1, extracted_compressibilities_1, rtol = 1e-1)
            assert np.allclose(expected_compressibilities_3, extracted_compressibilities_3, rtol = 1e-1)

        #Test that autocutting works properly... for now just take the intersection of both

    finally:
        shutil.rmtree(measurement_pathname)



def test_get_balanced_axial_squish_fitted_mu_and_T():
    _mu_and_T_fit_test_helper(analysis_functions.get_balanced_axial_squish_fitted_mu_and_T, 
                              data_fitting_functions.fit_li6_balanced_density, 
                              data_fitting_functions.fit_li6_balanced_density_with_prefactor)
    
def test_get_imbalanced_axial_squish_fitted_mu_and_T():
    _mu_and_T_fit_test_helper(analysis_functions.get_imbalanced_axial_squish_fitted_mu_and_T, 
                              data_fitting_functions.fit_li6_ideal_fermi_density, 
                              data_fitting_functions.fit_li6_ideal_fermi_density_with_prefactor)


def _mu_and_T_fit_test_helper(tested_analysis_function, fit_function, fit_function_with_prefactor):
    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "top_um_per_pixel":SQRT_2_DUMMY, 
        "axicon_diameter_pix":100,
        "axicon_tilt_deg":0.0,
        "axicon_side_aspect_ratio":1.0, 
        "axicon_side_angle_deg":0.0,
        "axial_trap_frequency_hz":E_DUMMY,
        "hybrid_trap_typical_length_pix":DEFAULT_ABS_SQUARE_WIDTH,
        "hybrid_trap_center_pix_polrot":255,
        "axial_gradient_Hz_per_um_V":SQRT_5_DUMMY
    }
    run_param_values = {
        "ImagFreq1":0.0, 
        "ImagFreq2":0.0,
        "Axial_Squish_Imaging_Grad_V":SQRT_7_DUMMY
    }
    hybrid_sample_image = get_hybrid_sample_absorption_image()
    hybrid_sample_image_stack = generate_image_stack_from_absorption(hybrid_sample_image)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = hybrid_sample_image_stack, 
                                                        run_param_values = run_param_values, experiment_param_values = hf_atom_density_experiment_param_values, 
                                                        ROI = DEFAULT_ABSORPTION_IMAGE_ROI, norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        my_measurement.analyze_runs(analysis_functions.get_atom_densities_top_abs, ("densities_1", "densities_3"))
        #Zero out densities 3 so that logic also works for imbalanced fitting
        my_run.analysis_results["densities_3"] = np.zeros(my_run.analysis_results["densities_1"].shape)
        #Obtain expected densities as cut by autocut
        potentials_1, densities_1, potentials_2, densities_2 = analysis_functions.get_axial_squish_densities_along_harmonic_axis(
                my_measurement, my_run, imaging_mode = "abs", autocut = True, return_positions = False,
                first_stored_density_name = "densities_1", second_stored_density_name = "densities_3"
            )
        #Fit will be terrible, but just check logic
        expected_fit_results_no_prefactor = fit_function(potentials_1, densities_1)
        no_prefactor_popt, no_prefactor_pcov = expected_fit_results_no_prefactor
        expected_mu_np, expected_T_np = no_prefactor_popt
        expected_mu_error_np, expected_T_error_np = np.sqrt(np.diag(no_prefactor_pcov))


        expected_fit_results_with_prefactor = fit_function_with_prefactor(potentials_1, densities_1)
        with_prefactor_popt, with_prefactor_pcov = expected_fit_results_with_prefactor
        expected_mu_wp, expected_T_wp, expected_prefactor = with_prefactor_popt
        expected_mu_error_wp, expected_T_error_wp, expected_prefactor_error = np.sqrt(np.diag(with_prefactor_pcov))

        #Now get the returns from the function itself
        extracted_mu_np, extracted_T_np, extracted_mu_error_np, extracted_T_error_np = tested_analysis_function(
            my_measurement, my_run, fit_prefactor = False, return_errors = True, first_stored_density_name = "densities_1", 
            second_stored_density_name = "densities_3"
        )
        assert np.isclose(extracted_mu_np, expected_mu_np)
        assert np.isclose(extracted_mu_error_np, expected_mu_error_np)
        assert np.isclose(extracted_T_np, expected_T_np)
        assert np.isclose(extracted_T_error_np, expected_T_error_np)
        (extracted_mu_wp, extracted_T_wp, extracted_prefactor, extracted_mu_error_wp, 
         extracted_T_error_wp, extracted_prefactor_error) = tested_analysis_function(
            my_measurement, my_run, fit_prefactor = True, return_errors = True, first_stored_density_name = "densities_1", 
            second_stored_density_name = "densities_3"
        )
        assert np.isclose(extracted_mu_wp, expected_mu_wp)
        assert np.isclose(extracted_mu_error_wp, expected_mu_error_wp)
        assert np.isclose(extracted_T_wp, expected_T_wp)
        assert np.isclose(extracted_T_error_wp, expected_T_error_wp)
        assert np.isclose(extracted_prefactor, expected_prefactor)
        assert np.isclose(extracted_prefactor_error, expected_prefactor_error)
    finally:
        shutil.rmtree(measurement_pathname)



def test_get_box_shake_fourier_amplitudes():
    hf_atom_density_experiment_param_values_no_tilt = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "top_um_per_pixel":SQRT_2_DUMMY, 
        "axicon_diameter_pix":100,
        "axicon_tilt_deg":0.0,
        "axicon_side_aspect_ratio":1.0, 
        "axicon_side_angle_deg":0.0,
        "hybrid_trap_typical_length_pix":DEFAULT_ABS_SQUARE_WIDTH,
        "axial_trap_frequency_hz":E_DUMMY
    }
    run_param_values = {
        "ImagFreq1":0.0, 
        "ImagFreq2":0.0
    }
    fourier_sample_image = get_fourier_component_sample_absorption_image()
    fourier_sample_image_stack = generate_image_stack_from_absorption(fourier_sample_image)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = fourier_sample_image_stack, 
                                                        run_param_values = run_param_values, 
                                                        experiment_param_values = hf_atom_density_experiment_param_values_no_tilt, 
                                                        ROI = DEFAULT_ABSORPTION_IMAGE_CLOSE_ROI, norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        #Test and compare gettting vs. storing the densities
        my_measurement.analyze_runs(analysis_functions.get_atom_densities_top_abs, ("densities_1", "densities_3"))
        fourier_components_stored_densities = analysis_functions.get_box_shake_fourier_amplitudes(my_measurement, my_run, 
                                                                            first_stored_density_name = "densities_1", 
                                                                            second_stored_density_name = "densities_3", 
                                                                            order = FOURIER_SAMPLE_ORDER)
        fourier_components_unstored_densities = analysis_functions.get_box_shake_fourier_amplitudes(my_measurement, my_run, 
                                                                            imaging_mode = "abs", order = FOURIER_SAMPLE_ORDER)
        assert np.all(np.isclose(fourier_components_stored_densities, fourier_components_unstored_densities))
        fourier_component_1, fourier_component_2 = fourier_components_stored_densities 
        assert np.isclose(fourier_component_1, fourier_component_2)
        top_um_per_pixel = hf_atom_density_experiment_param_values_no_tilt["top_um_per_pixel"]
        expected_fourier_amplitude = FOURIER_SAMPLE_AMPLITUDE *  (1.0 / li_6_res_cross_section) * DEFAULT_ABS_SQUARE_WIDTH * top_um_per_pixel
        assert np.isclose(fourier_component_1, expected_fourier_amplitude, rtol = 5e-3)
        fourier_components_with_phase = analysis_functions.get_box_shake_fourier_amplitudes(my_measurement, my_run, 
                                                                                first_stored_density_name = "densities_1", 
                                                                                second_stored_density_name = "densities_3", 
                                                                                return_phases = True)
        amp_1, phase_1, amp_2, phase_2 = fourier_components_with_phase 
        assert np.isclose(amp_1, fourier_component_1)
        assert np.isclose(amp_1, amp_2) 
        assert np.isclose(phase_1, phase_2) 
        assert np.isclose(amp_1, expected_fourier_amplitude, rtol = 5e-3)
        #Also check phase information
        assert np.isclose(0.0, phase_1, atol = 3e-2)
    finally:
        shutil.rmtree(measurement_pathname)

    #Now test with autocutting in a tilted box...
    SAMPLE_AXICON_TILT_DEG = 5.0 
    hf_atom_density_experiment_param_values_with_tilt = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "top_um_per_pixel":SQRT_2_DUMMY, 
        "axicon_diameter_pix":DEFAULT_ABS_SQUARE_WIDTH,
        "box_length_pix":DEFAULT_ABS_SQUARE_WIDTH,
        "axicon_tilt_deg":SAMPLE_AXICON_TILT_DEG,
        "axicon_side_aspect_ratio":1.0, 
        "axicon_side_angle_deg":0.0,
        "hybrid_trap_typical_length_pix":DEFAULT_ABS_SQUARE_WIDTH,
        "axial_trap_frequency_hz":E_DUMMY
    }
    tilted_fourier_image = scipy.ndimage.rotate(fourier_sample_image, -SAMPLE_AXICON_TILT_DEG, reshape = False, cval = 1.0)
    tilted_fourier_sample_image_stack = generate_image_stack_from_absorption(tilted_fourier_image)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = tilted_fourier_sample_image_stack, 
                                                        run_param_values = run_param_values, 
                                                        experiment_param_values = hf_atom_density_experiment_param_values_with_tilt, 
                                                        ROI = DEFAULT_ABSORPTION_IMAGE_ROI, norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        #Test with autocut        
        fourier_components_tilted_autocut = analysis_functions.get_box_shake_fourier_amplitudes(my_measurement, my_run, 
                                                                            imaging_mode = "abs", order = FOURIER_SAMPLE_ORDER, 
                                                                            autocut = True)
        fourier_component_1, fourier_component_2 = fourier_components_tilted_autocut 
        assert np.isclose(fourier_component_1, fourier_component_2)
        top_um_per_pixel = hf_atom_density_experiment_param_values_with_tilt["top_um_per_pixel"]
        expected_fourier_amplitude = FOURIER_SAMPLE_AMPLITUDE *  (1.0 / li_6_res_cross_section) * DEFAULT_ABS_SQUARE_WIDTH * top_um_per_pixel
        #Tolerance is pretty bad because of the rotation... 
        assert np.isclose(fourier_component_1, expected_fourier_amplitude, rtol = 3e-2)
        fourier_components_with_phase_tilted_autocut = analysis_functions.get_box_shake_fourier_amplitudes(my_measurement, my_run, 
                                                                            imaging_mode = "abs", order = FOURIER_SAMPLE_ORDER, 
                                                                            autocut = True, return_phases = True)
        amp_1, phase_1, amp_2, phase_2 = fourier_components_with_phase_tilted_autocut
        assert np.isclose(amp_1, fourier_component_1)
        assert np.isclose(amp_1, amp_2)
        assert np.isclose(phase_1, phase_2) 
        assert np.isclose(amp_1, expected_fourier_amplitude, rtol = 3e-2)
        #Also check phase information
        assert np.isclose(0.0, phase_1, atol = 2e-2)
    finally:
        shutil.rmtree(measurement_pathname)

def test_get_box_in_situ_fermi_energies_from_counts():
    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_unitarity_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "top_um_per_pixel":SQRT_2_DUMMY, 
        "axicon_diameter_pix":100,
        "box_length_pix": 141,
        "axicon_tilt_deg":5.0,
        "axicon_side_aspect_ratio":1.0, 
        "axicon_side_angle_deg":0.0,
    }
    run_param_values = {
        "ImagFreq1":0.0, 
        "ImagFreq2":0.0
    }
    default_absorption_image = get_default_absorption_image() 
    default_image_stack = generate_image_stack_from_absorption(default_absorption_image) 
    default_absorption_image_cropped = get_default_absorption_image(crop_to_roi = True)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = default_image_stack, 
                                                        run_param_values = run_param_values, experiment_param_values = hf_atom_density_experiment_param_values, 
                                                        ROI = DEFAULT_ABSORPTION_IMAGE_ROI, norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        #Calculate the expected Fermi energies; remember, this only tests piping, not the logic for E_F calculation
        um_per_pixel = hf_atom_density_experiment_param_values["top_um_per_pixel"]
        expected_atom_densities_cropped = -np.log(default_absorption_image_cropped) / li_6_res_cross_section
        expected_atom_counts = np.square(um_per_pixel) * np.sum(expected_atom_densities_cropped)

        box_radius_um = um_per_pixel * hf_atom_density_experiment_param_values["axicon_diameter_pix"] / 2
        box_length_um = um_per_pixel * hf_atom_density_experiment_param_values["box_length_pix"]
        box_cross_section_um = analysis_functions.get_hybrid_cross_section_um(my_measurement, axis = "axicon")
        expected_fermi_energy = science_functions.get_box_fermi_energy_from_counts(expected_atom_counts, box_cross_section_um, box_length_um)
        
        #Now get them from the analysis function...
        #First use unstored densities
        fermi_energy_1, fermi_energy_2 = analysis_functions.get_box_in_situ_fermi_energies_from_counts(
            my_measurement, my_run, 
            imaging_mode = "abs"
        )
        assert np.isclose(fermi_energy_1, fermi_energy_2)
        assert np.isclose(fermi_energy_1, expected_fermi_energy, rtol = 1e-3)
        assert np.isclose(fermi_energy_2, expected_fermi_energy, rtol = 1e-3)
        #Test and compare gettting vs. storing the densities 
        my_measurement.analyze_runs(analysis_functions.get_atom_densities_top_abs, ("densities_1", "densities_3"))
        fermi_energy_1_stored, fermi_energy_2_stored = analysis_functions.get_box_in_situ_fermi_energies_from_counts(
            my_measurement, my_run, first_stored_density_name = "densities_1", second_stored_density_name = "densities_3"
        )
        assert np.isclose(fermi_energy_1_stored, fermi_energy_2_stored)
        assert np.isclose(fermi_energy_1_stored, expected_fermi_energy, rtol = 1e-3)
    finally:
        shutil.rmtree(measurement_pathname)


def test_get_rapid_ramp_densities_along_harmonic_axis():
    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_rr_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "top_um_per_pixel":SQRT_2_DUMMY, 
        "rr_tilt_deg":-5.0,
    }
    run_param_values = {
        "ImagFreq1":0.0, 
        "ImagFreq2":0.0
    }
    rotation_angle_deg = hf_atom_density_experiment_param_values["rr_tilt_deg"]
    default_absorption_image = get_default_absorption_image()
    rotated_default_absorption_image = scipy.ndimage.rotate(default_absorption_image, -rotation_angle_deg, reshape = False)
    rotated_default_image_stack = generate_image_stack_from_absorption(rotated_default_absorption_image)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = rotated_default_image_stack, 
                                                        run_param_values = run_param_values, experiment_param_values = hf_atom_density_experiment_param_values, 
                                                        ROI = DEFAULT_ABSORPTION_IMAGE_ROI, norm_box = DEFAULT_ABSORPTION_IMAGE_NORM_BOX)
        #Calculate the expected profile, which should be un-rotated compared to the previous
        um_per_pixel = hf_atom_density_experiment_param_values["top_um_per_pixel"]
        rr_density_pre_rotation = -np.log(rotated_default_absorption_image) / li_6_res_cross_section
        roi_xmin, roi_ymin, roi_xmax, roi_ymax = DEFAULT_ABSORPTION_IMAGE_ROI
        rr_density_pre_rotation_cropped = rr_density_pre_rotation[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
        rr_density_expected = scipy.ndimage.rotate(rr_density_pre_rotation_cropped, rotation_angle_deg, reshape = False)
        rr_integrated_density_expected = np.sum(rr_density_expected, axis = 1) * um_per_pixel
        #Test getting densities both by storing and by re-processing
        rr_integrated_densities = analysis_functions.get_rapid_ramp_densities_along_harmonic_axis(my_measurement, my_run, imaging_mode = "abs", 
                                                                                                  b_field_condition = "rapid_ramp")
        assert np.all(np.isclose(rr_integrated_densities[0], rr_integrated_densities[1]))
        my_measurement.analyze_runs(analysis_functions.get_atom_densities_top_abs, ("densities_1", "densities_3"), 
                                    fun_kwargs = {
                                        "b_field_condition":"rapid_ramp"
                                    })
        rr_integrated_densities_stored = analysis_functions.get_rapid_ramp_densities_along_harmonic_axis(my_measurement, my_run,
                                                                        first_stored_density_name = "densities_1",
                                                                        second_stored_density_name = "densities_3")
        assert np.all(np.isclose(rr_integrated_densities, rr_integrated_densities_stored))
        #Rotating and rerotating causes a Gibbs phenomenon on our square profile; the tolerances are accordingly rather high...
        assert np.all(np.isclose(rr_integrated_densities[0], rr_integrated_density_expected, rtol = 1e-3, atol = 1e-1))
    finally:
        shutil.rmtree(measurement_pathname)

def test_get_rr_condensate_fractions_fit():
    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_rr_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "top_um_per_pixel":SQRT_2_DUMMY, 
        "rr_tilt_deg":-5.0,
    }
    run_param_values = {
        "ImagFreq1":0.0, 
        "ImagFreq2":0.0
    }
    rotation_angle_deg = hf_atom_density_experiment_param_values["rr_tilt_deg"]
    rr_condensate_sample_absorption_image = get_rr_condensate_sample_absorption_image()
    rotated_rr_condensate_sample_absorption_image = scipy.ndimage.rotate(rr_condensate_sample_absorption_image, -rotation_angle_deg, reshape = False)
    rotated_default_image_stack = generate_image_stack_from_absorption(rotated_rr_condensate_sample_absorption_image)
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = rotated_default_image_stack, 
                                                        run_param_values = run_param_values, experiment_param_values = hf_atom_density_experiment_param_values, 
                                                        ROI = EXPANDED_ABSORPTION_IMAGE_ROI, norm_box = EXPANDED_ABSORPTION_IMAGE_NORM_BOX)
        rr_condensate_fractions = analysis_functions.get_rr_condensate_fractions_fit(my_measurement, my_run)
        rr_condensate_fraction_1, rr_condensate_fraction_2 = rr_condensate_fractions 
        assert np.isclose(rr_condensate_fraction_1, rr_condensate_fraction_2)
        expected_condensate_integral = data_fitting_functions.one_d_condensate_integral(DEFAULT_ABS_SQUARE_CENTER_INDICES, RR_CONDENSATE_WIDTH, RR_CONDENSATE_MAGNITUDE)
        expected_thermal_integral = data_fitting_functions.thermal_bose_integral(DEFAULT_ABS_SQUARE_CENTER_INDICES, RR_THERMAL_WIDTH, RR_THERMAL_MAGNITUDE)
        expected_condensate_fraction = expected_condensate_integral / (expected_thermal_integral + expected_condensate_integral)
        assert np.isclose(rr_condensate_fraction_1, expected_condensate_fraction, rtol = 1e-3)
    finally:
        shutil.rmtree(measurement_pathname)

def test_get_rr_condensate_fractions_box():
    hf_atom_density_experiment_param_values = {
        "state_1_unitarity_res_freq_MHz": 0.0,
        "state_3_unitarity_res_freq_MHz":0.0,
        "hf_lock_rr_resonance_value":0.0,
        "hf_lock_setpoint":0.0,
        "hf_lock_frequency_multiplier":1.0,
        "li_top_sigma_multiplier":1.0,
        "li_hf_freq_multiplier":1.0,
        "top_um_per_pixel":SQRT_2_DUMMY, 
        "rr_tilt_deg":-5.0,
    }
    run_param_values = {
        "ImagFreq1":0.0, 
        "ImagFreq2":0.0
    }
    rotation_angle_deg = hf_atom_density_experiment_param_values["rr_tilt_deg"]
    rr_condensate_sample_absorption_image = get_rr_condensate_sample_absorption_image()
    rotated_rr_condensate_sample_absorption_image = scipy.ndimage.rotate(rr_condensate_sample_absorption_image, -rotation_angle_deg, reshape = False)
    rotated_default_image_stack = generate_image_stack_from_absorption(rotated_rr_condensate_sample_absorption_image)
    rr_roi = [100, 256 - RR_CONDENSATE_WIDTH, 400, 256 + RR_CONDENSATE_WIDTH]
    shifted_rr_roi = [100, 256 - 3*RR_CONDENSATE_WIDTH, 400, 256 - RR_CONDENSATE_WIDTH]
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double", image_stack = rotated_default_image_stack, 
                                                        run_param_values = run_param_values, experiment_param_values = hf_atom_density_experiment_param_values, 
                                                        ROI = EXPANDED_ABSORPTION_IMAGE_ROI, norm_box = EXPANDED_ABSORPTION_IMAGE_NORM_BOX)
        my_measurement.set_box("rr_condensate_roi", box_coordinates = rr_roi)
        expected_density = -np.log(rr_condensate_sample_absorption_image) / li_6_res_cross_section
        rr_roi_xmin, rr_roi_ymin, rr_roi_xmax, rr_roi_ymax = rr_roi
        rr_roi_sum = np.sum(expected_density[rr_roi_ymin:rr_roi_ymax, rr_roi_xmin:rr_roi_xmax])
        shifted_rr_roi_xmin, shifted_rr_roi_ymin, shifted_rr_roi_xmax, shifted_rr_roi_ymax = shifted_rr_roi 
        shifted_rr_roi_sum = np.sum(expected_density[shifted_rr_roi_ymin:shifted_rr_roi_ymax, shifted_rr_roi_xmin:shifted_rr_roi_xmax])
        shift_subtracted_roi_sum = rr_roi_sum - shifted_rr_roi_sum 
        expanded_roi_xmin, expanded_roi_ymin, expanded_roi_xmax, expanded_roi_ymax = EXPANDED_ABSORPTION_IMAGE_ROI
        expanded_roi_sum = np.sum(expected_density[expanded_roi_ymin:expanded_roi_ymax, expanded_roi_xmin:expanded_roi_xmax])
        expected_rr_box_condensate_fraction = shift_subtracted_roi_sum / expanded_roi_sum 
        #Now get it from analysis functions
        rr_box_condensate_fractions = analysis_functions.get_rr_condensate_fractions_box(my_measurement, my_run)
        rr_box_condensate_fraction_1, rr_box_condensate_fraction_2 = rr_box_condensate_fractions 
        assert np.isclose(rr_box_condensate_fraction_1, rr_box_condensate_fraction_2)
        #And let's see if the expected matches the returned 
        assert np.isclose(rr_box_condensate_fraction_1, expected_rr_box_condensate_fraction, rtol = 1e-3)
    finally:
        shutil.rmtree(measurement_pathname)


def test_get_uniform_reshaped_density():
    try:
        measurement_pathname, my_measurement, _ = create_measurement("top_double")
        def create_fake_run(value):
            parameters_dict = {"id":value}
            num_points = (value + 1) * (value + 2)
            analysis_dict = {"foo":np.arange(num_points).reshape((value + 1, value + 2))}
            return measurement.Run(value, None, parameters_dict, analysis_results = analysis_dict, connected_mode = False)
        NUM_RUNS = 5
        for i in range(NUM_RUNS):
            my_measurement.runs_dict[i] = create_fake_run(i)
        ARB_RUN_VALUE = 2
        arb_run = my_measurement.runs_dict[ARB_RUN_VALUE] 

        arb_run_original_array = np.arange((ARB_RUN_VALUE + 1) * (ARB_RUN_VALUE + 2)).reshape((ARB_RUN_VALUE + 1, ARB_RUN_VALUE + 2))
        expected_increment = ARB_RUN_VALUE
        expected_lower_increment_sym = expected_increment // 2
        expected_upper_increment_sym = expected_increment - expected_lower_increment_sym

        EXPECTED_CROPPED_SHAPE = (1, 2)

        cropped_arb_density_sym = analysis_functions.get_uniform_reshaped_density(my_measurement, arb_run, 
                                                        stored_density_name = "foo", crop_not_pad = True, crop_or_pad_position = "sym")
        expected_sym_crop = arb_run_original_array[expected_lower_increment_sym:expected_lower_increment_sym + EXPECTED_CROPPED_SHAPE[0], 
                                                   expected_lower_increment_sym:expected_lower_increment_sym + EXPECTED_CROPPED_SHAPE[1]]
        assert np.all(cropped_arb_density_sym == expected_sym_crop)

        cropped_arb_density_lower = analysis_functions.get_uniform_reshaped_density(my_measurement, arb_run, 
                                                                            stored_density_name = "foo", crop_not_pad = True, crop_or_pad_position = "lower")
        expected_lower_crop = arb_run_original_array[expected_increment:, expected_increment:] 
        assert np.all(cropped_arb_density_lower == expected_lower_crop)

        cropped_arb_density_upper = analysis_functions.get_uniform_reshaped_density(my_measurement, arb_run, 
                                                                            stored_density_name = "foo", crop_not_pad = True, crop_or_pad_position = "upper")
        expected_upper_crop = arb_run_original_array[:-expected_increment, :-expected_increment] 
        assert np.all(cropped_arb_density_upper == expected_upper_crop)

        my_measurement.measurement_analysis_results.pop("foo_uniform_reshape_dimensions")

        padded_arb_density_sym = analysis_functions.get_uniform_reshaped_density(my_measurement, arb_run, 
                                                                            stored_density_name = "foo", crop_not_pad = False, 
                                                                            crop_or_pad_position = "sym")
        expected_sym_padded = np.pad(arb_run_original_array, ((expected_lower_increment_sym, expected_upper_increment_sym), 
                                                            (expected_lower_increment_sym, expected_upper_increment_sym)))
        assert np.all(padded_arb_density_sym == expected_sym_padded)

        padded_arb_density_lower = analysis_functions.get_uniform_reshaped_density(my_measurement, arb_run, 
                                                                            stored_density_name = "foo", crop_not_pad = False, 
                                                                            crop_or_pad_position = "lower")
        expected_lower_padded = np.pad(arb_run_original_array, ((expected_increment, 0), 
                                                            (expected_increment, 0)))
        assert np.all(padded_arb_density_lower == expected_lower_padded)

        padded_arb_density_upper = analysis_functions.get_uniform_reshaped_density(my_measurement, arb_run, 
                                                                            stored_density_name = "foo", crop_not_pad = False, 
                                                                            crop_or_pad_position = "upper")
        expected_upper_padded = np.pad(arb_run_original_array, ((0, expected_increment), 
                                                                (0, expected_increment)))
        assert np.all(padded_arb_density_upper == expected_upper_padded)
        


    finally:
        shutil.rmtree(measurement_pathname)

def test_get_uniform_density_reshape_dimensions():
    try:
        measurement_pathname, my_measurement, _ = create_measurement("top_double")
        def create_fake_run(value):
            parameters_dict = {"id":value}
            analysis_dict = {"foo":np.ones((value + 1, value + 2))}
            return measurement.Run(value, None, parameters_dict, analysis_results = analysis_dict, connected_mode = False)
        NUM_RUNS = 4
        for i in range(NUM_RUNS):
            my_measurement.runs_dict[i] = create_fake_run(i) 
        #Manually remove a density to ensure correct ignoring
        my_measurement.runs_dict[2].analysis_results.pop("foo")
        cropped_dimensions = analysis_functions.get_uniform_density_reshape_dimensions(my_measurement, "foo", True)
        assert cropped_dimensions == (1, 2)
        padded_dimensions = analysis_functions.get_uniform_density_reshape_dimensions(my_measurement, "foo", False)
        assert padded_dimensions == (NUM_RUNS, NUM_RUNS + 1)
    finally:
        shutil.rmtree(measurement_pathname)


def test_get_saturation_counts_top():
    experiment_param_values = {
        "li_top_sigma_multiplier":L33T_DUMMY,
        "top_um_per_pixel":SQRT_2_DUMMY, 
        "top_camera_quantum_efficiency":1.0 / np.e,
        "top_camera_counts_per_photoelectron":1.0 / np.pi,
        "top_camera_post_atom_photon_transmission":1.0 / np.sqrt(5),
        "top_camera_saturation_ramsey_fudge":1.337
    }

    run_param_values = {
        "ImageTime":42.7
    }

    imaging_time_us = run_param_values["ImageTime"] 
    imaging_time = imaging_time_us * 1e-6
    sigma_multiplier = experiment_param_values["li_top_sigma_multiplier"]
    um_per_pixel = experiment_param_values["top_um_per_pixel"] 
    m_per_pixel = um_per_pixel * 1e-6
    quantum_efficiency = experiment_param_values["top_camera_quantum_efficiency"]
    counts_per_photoelectron = experiment_param_values["top_camera_counts_per_photoelectron"]
    photon_transmission = experiment_param_values["top_camera_post_atom_photon_transmission"] 
    saturation_ramsey_fudge = experiment_param_values["top_camera_saturation_ramsey_fudge"] 
    linewidth_MHz = li_6_linewidth 
    gamma = 2 * np.pi * linewidth_MHz * 1e6
    cross_section_um = li_6_res_cross_section
    cross_section = cross_section_um * 1e-12


    #Independently derived without reference to code
    #Note that linewidth is in MHz, imaging time in 
    expected_saturation_counts = (
        (np.square(m_per_pixel) * quantum_efficiency * photon_transmission * imaging_time* counts_per_photoelectron * gamma) / 
        (2 * sigma_multiplier * cross_section)
    )
    try:
        measurement_pathname, my_measurement, my_run = create_measurement("top_double",
                                                        run_param_values = run_param_values, experiment_param_values = experiment_param_values, 
                                                        )
        saturation_counts = analysis_functions.get_saturation_counts_top(my_measurement, my_run, apply_ramsey_fudge = False) 
        assert np.isclose(expected_saturation_counts, saturation_counts)
        saturation_counts_fudged = analysis_functions.get_saturation_counts_top(my_measurement, my_run, apply_ramsey_fudge = True) 
        #The ramsey fudge multiplies the saturation at fixed counts, so if the saturation counts are to be determined, they get divided by the fudge
        expected_saturation_counts_fudged = expected_saturation_counts / saturation_ramsey_fudge
        assert np.isclose(expected_saturation_counts_fudged, saturation_counts_fudged)
    finally:
        shutil.rmtree(measurement_pathname)

def test_get_hybrid_cross_section_um():
    SAMPLE_SIDE_ANGLE_DEG = 45
    SAMPLE_SIDE_ASPECT_RATIO = 2 
    SAMPLE_TOP_RADIUS_PIX = 100
    SAMPLE_TOP_UM_PER_PIXEL = 3.14
    SAMPLE_AXICON_ANGLE_DEG = 10
    dummy_measurement = DummyMeasurement() 
    dummy_measurement.experiment_parameters = {
        "axicon_diameter_pix":SAMPLE_TOP_RADIUS_PIX * 2,
        "top_um_per_pixel":SAMPLE_TOP_UM_PER_PIXEL,
        "axicon_side_angle_deg":SAMPLE_SIDE_ANGLE_DEG, 
        "axicon_side_aspect_ratio":SAMPLE_SIDE_ASPECT_RATIO,
        "axicon_tilt_deg":SAMPLE_AXICON_ANGLE_DEG
    }

    #From hand-evaluation of the formula
    EXPECTED_CROSS_SECTION_UM2_AXICON = 247798.775431
    cross_section_axicon = analysis_functions.get_hybrid_cross_section_um(dummy_measurement, axis = "axicon")
    assert np.isclose(cross_section_axicon, EXPECTED_CROSS_SECTION_UM2_AXICON)
    expected_cross_section_um2_harmonic = EXPECTED_CROSS_SECTION_UM2_AXICON / np.cos(np.deg2rad(SAMPLE_AXICON_ANGLE_DEG))
    cross_section_harmonic = analysis_functions.get_hybrid_cross_section_um(dummy_measurement, axis = "harmonic")
    assert np.isclose(cross_section_harmonic, expected_cross_section_um2_harmonic)




def test_identical_parameters_run_hash_function():
    def create_fake_run(value):
        parameters_dict = {"id":value, "runtime":"00:00:00", "mod":value % 2, "foo":"bar"}
        return measurement.Run(value, None, parameters_dict, connected_mode = False)
    hash_even = analysis_functions.identical_parameters_run_hash_function(create_fake_run(0)) 
    hash_odd = analysis_functions.identical_parameters_run_hash_function(create_fake_run(1))
    for i in range(10):
        fake_run = create_fake_run(i)
        run_hash = analysis_functions.identical_parameters_run_hash_function(fake_run)
        if i % 2 == 0:
            matching_hash = hash_even 
            other_hash = hash_odd
        else:
            matching_hash = hash_odd 
            other_hash = hash_even
        assert run_hash == matching_hash 
        assert run_hash != other_hash


def test_arbitrary_function_identical_runs_run_combine_function_factory():
    def create_fake_run(value):
        parameters_dict = {"id":value, "foo":1}
        analysis_dict = {"val":value, "double_val":2 * value}
        return measurement.Run(value, None, parameters_dict, analysis_results = analysis_dict, connected_mode = False)
    run_list = [create_fake_run(i) for i in range(10)]
    run_combine_function_val = analysis_functions.average_results_identical_runs_run_combine_function_factory("val", np.average)
    returned_run = run_combine_function_val(run_list)
    expected_id_string = ",".join([str(i) for i in range(10)])
    assert returned_run.parameters["id"] == expected_id_string
    assert returned_run.parameters["foo"] == 1
    assert returned_run.analysis_results["val"] == np.average(range(10)) 
    assert not "double_val" in returned_run.analysis_results

    run_combine_function_both = analysis_functions.average_results_identical_runs_run_combine_function_factory(("val", "double_val"), np.average)
    returned_run = run_combine_function_both(run_list)
    assert returned_run.parameters["id"] == expected_id_string 
    assert returned_run.parameters["foo"] == 1 
    assert returned_run.analysis_results["val"] == np.average(range(10)) 
    assert returned_run.analysis_results["double_val"] == 2 * np.average(range(10))


def test_average_densities_13_run_combine_function():
    def create_fake_run(value):
        parameters_dict = {"id":value}
        analysis_dict = {"densities_1":value * np.arange(100).reshape((10, 10)), "densities_3":-value * np.arange(100).reshape((10, 10))}
        return measurement.Run(value, None, parameters_dict, analysis_results = analysis_dict, connected_mode = False)
    run_list = [create_fake_run(i) for i in range(2)]
    expected_densities_1 = 0.5 * np.arange(100).reshape((10, 10))
    expected_densities_3 = -0.5 * np.arange(100).reshape((10, 10))
    returned_run = analysis_functions.average_densities_13_run_combine_function(run_list)
    assert np.allclose(returned_run.analysis_results["densities_1"], expected_densities_1) 
    assert np.allclose(returned_run.analysis_results["densities_3"], expected_densities_3)


def create_measurement(type_name, image_stack = None, run_param_values= None, experiment_param_values = None, ROI = None, norm_box = None):
    if image_stack is None:
        default_abs_image = get_default_absorption_image()
        image_stack = generate_image_stack_from_absorption(default_abs_image)
    if experiment_param_values is None:
        experiment_param_values = {} 
    if run_param_values is None:
        run_param_values = {}         
    measurement_pathname = create_dummy_measurement_folder(image_stack, run_param_values,
                                experiment_param_values, type_name)
    my_measurement = measurement.Measurement(measurement_directory_path = measurement_pathname, 
                                    imaging_type = type_name)
    if not ROI is None:
        my_measurement.set_ROI(box_coordinates = ROI)
    if not norm_box is None:
        my_measurement.set_norm_box(box_coordinates = norm_box)
    for run_id in my_measurement.runs_dict:
        my_run = my_measurement.runs_dict[run_id] 
        break
    return (measurement_pathname, my_measurement, my_run)


#Generate a test image, suitable for use in most analysis functions
#The image pattern is a square, with -ln(abs) = 1 for a grid of pixels centered 
#on the origin and -ln(abs) = 0 for all others.
def get_default_absorption_image(crop_to_roi = False):
    center_y_index, center_x_index = DEFAULT_ABS_SQUARE_CENTER_INDICES
    y_indices, x_indices = np.indices(DEFAULT_ABS_IMAGE_SHAPE)
    default_absorption_image = np.where(
        np.logical_and(
            np.abs(y_indices - center_y_index) < (DEFAULT_ABS_SQUARE_WIDTH + 1) //2, 
            np.abs(x_indices - center_x_index) < (DEFAULT_ABS_SQUARE_WIDTH + 1) // 2
        ),
        1.0/np.e, 
        1.0
    )
    if not crop_to_roi:
        return default_absorption_image
    else:
        roi_xmin, roi_ymin, roi_xmax, roi_ymax = DEFAULT_ABSORPTION_IMAGE_ROI
        return default_absorption_image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
    
def generate_default_image_density_pattern(crop_to_roi = False, density_value = 0.0):
    center_y_index, center_x_index = DEFAULT_ABS_SQUARE_CENTER_INDICES
    y_indices, x_indices = np.indices(DEFAULT_ABS_IMAGE_SHAPE)
    default_density_pattern = np.where(
        np.logical_and(
            np.abs(y_indices - center_y_index) < (DEFAULT_ABS_SQUARE_WIDTH + 1) //2, 
            np.abs(x_indices - center_x_index) < (DEFAULT_ABS_SQUARE_WIDTH + 1) // 2
        ),
        density_value, 
        0.0
    )
    if not crop_to_roi:
        return default_density_pattern
    else:
        roi_xmin, roi_ymin, roi_xmax, roi_ymax = DEFAULT_ABSORPTION_IMAGE_ROI
        return default_density_pattern[roi_ymin:roi_ymax, roi_xmin:roi_xmax]    

def get_box_autocut_absorption_image(crop_to_roi = False):
    center_y_index, center_x_index = DEFAULT_ABS_SQUARE_CENTER_INDICES
    y_indices, x_indices = np.indices(DEFAULT_ABS_IMAGE_SHAPE)
    box_radius = (DEFAULT_ABS_SQUARE_WIDTH + 1) // 2
    box_autocut_absorption_image = np.where(
        np.logical_and(
            np.abs(y_indices - center_y_index) < box_radius,
            np.abs(x_indices - center_x_index) < box_radius
        ), 
        np.exp(-np.sqrt(1 - np.square((x_indices - center_x_index) / box_radius))),
        1.0
    )
    if not crop_to_roi:
        return box_autocut_absorption_image 
    else:
        roi_xmin, roi_ymin, roi_xmax, roi_ymax = DEFAULT_ABSORPTION_IMAGE_ROI
        return box_autocut_absorption_image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]

def get_rotated_rectangle_absorption_image(angle, crop_to_roi = False):
    center_y_index, center_x_index = DEFAULT_ABS_SQUARE_CENTER_INDICES
    y_indices, x_indices = np.indices(DEFAULT_ABS_IMAGE_SHAPE)
    base_half_width = (DEFAULT_ABS_SQUARE_WIDTH + 1) // 2
    base_rectangle = np.where(
        np.logical_and(
            np.abs(y_indices - center_y_index) < base_half_width,
            np.abs(x_indices - center_x_index) < base_half_width / 2
        ),
        np.exp(-np.sqrt(1 - np.square((x_indices - center_x_index) / (base_half_width / 2)))),
        1.0
    )
    #Now rotate
    rotated_rectangle = scipy.ndimage.rotate(base_rectangle, angle, reshape = False, cval = 1.0)
    if not crop_to_roi:
        return rotated_rectangle 
    else:
        roi_xmin, roi_ymin, roi_xmax, roi_ymax = DEFAULT_ABSORPTION_IMAGE_ROI
        return rotated_rectangle[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
    
def get_offset_box_absorption_image(crop_to_roi = False):
    center_y_index, center_x_index = np.array(DEFAULT_ABS_SQUARE_CENTER_INDICES) + np.array([DENSITY_COM_OFFSET, -DENSITY_COM_OFFSET])
    y_indices, x_indices = np.indices(DEFAULT_ABS_IMAGE_SHAPE)
    box_radius = (DEFAULT_ABS_SQUARE_WIDTH + 1) // 2
    box_autocut_absorption_image = np.where(
        np.logical_and(
            np.abs(y_indices - center_y_index) < box_radius,
            np.abs(x_indices - center_x_index) < box_radius
        ), 
        np.exp(-np.sqrt(1 - np.square((x_indices - center_x_index) / box_radius))),
        1.0
    )
    if not crop_to_roi:
        return box_autocut_absorption_image 
    else:
        roi_xmin, roi_ymin, roi_xmax, roi_ymax = DEFAULT_ABSORPTION_IMAGE_ROI
        return box_autocut_absorption_image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]

def get_hybrid_sample_absorption_image(crop_to_roi = False):
    center_y_index, center_x_index = DEFAULT_ABS_SQUARE_CENTER_INDICES
    y_indices, x_indices = np.indices(DEFAULT_ABS_IMAGE_SHAPE)
    box_radius = (DEFAULT_ABS_SQUARE_WIDTH + 1) // 2
    hybrid_sample_absorption_image = np.where(
        np.logical_and(
            np.abs(y_indices - center_y_index) < box_radius,
            np.abs(x_indices - center_x_index) < box_radius
        ), 
        np.exp(-(1.0 - np.square((y_indices - center_y_index)/box_radius))),
        1.0
    )
    if not crop_to_roi:
        return hybrid_sample_absorption_image 
    else:
        roi_xmin, roi_ymin, roi_xmax, roi_ymax = DEFAULT_ABSORPTION_IMAGE_ROI
        return hybrid_sample_absorption_image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
    
def get_hybrid_sample_absorption_image_norm_pressure(crop_to_roi = False):
    center_y_index, center_x_index = DEFAULT_ABS_SQUARE_CENTER_INDICES
    y_indices, x_indices = np.indices(DEFAULT_ABS_IMAGE_SHAPE)
    box_radius = (DEFAULT_ABS_SQUARE_WIDTH + 1) // 2
    def unit_lorentzian(index, center, radius):
        return 1.0 / (1 + np.square(index - center) / np.square(radius))
    hybrid_sample_absorption_image = np.where(
        np.logical_and(
            np.abs(y_indices - center_y_index) < box_radius,
            np.abs(x_indices - center_x_index) < box_radius
        ), 
        np.exp(-unit_lorentzian(y_indices, center_y_index, box_radius / 3.0)),
        1.0
    )
    if not crop_to_roi:
        return hybrid_sample_absorption_image 
    else:
        roi_xmin, roi_ymin, roi_xmax, roi_ymax = DEFAULT_ABSORPTION_IMAGE_ROI
        return hybrid_sample_absorption_image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]

    
def get_rr_condensate_sample_absorption_image(crop_to_roi = False):
    center_y_index, center_x_index = DEFAULT_ABS_SQUARE_CENTER_INDICES
    y_indices, x_indices = np.indices(DEFAULT_ABS_IMAGE_SHAPE)
    box_radius = (DEFAULT_ABS_SQUARE_WIDTH + 1) // 2
    condensate_sample_optical_densities = np.where(
        np.abs(x_indices - center_x_index) < box_radius, 
        data_fitting_functions.one_d_condensate_function(y_indices, center_y_index, RR_CONDENSATE_WIDTH, RR_CONDENSATE_MAGNITUDE), 
        0.0
    )
    thermal_sample_optical_densities = np.where(
        np.abs(x_indices - center_x_index) < box_radius, 
        data_fitting_functions.thermal_bose_function(y_indices, center_y_index, RR_THERMAL_WIDTH, RR_THERMAL_MAGNITUDE),
        0.0
    )
    overall_optical_densities = condensate_sample_optical_densities + thermal_sample_optical_densities
    condensate_sample_absorption_image = np.exp(-overall_optical_densities)
    if not crop_to_roi:
        return condensate_sample_absorption_image 
    else:
        roi_xmin, roi_ymin, roi_xmax, roi_ymax = DEFAULT_ABSORPTION_IMAGE_ROI
        return condensate_sample_absorption_image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]

#This one uniquely uses a closer ROI than the others, so that the Fourier components are as one would expect
def get_fourier_component_sample_absorption_image(crop_to_roi = False):
    center_y_index, center_x_index = DEFAULT_ABS_SQUARE_CENTER_INDICES
    y_indices, x_indices = np.indices(DEFAULT_ABS_IMAGE_SHAPE)
    box_radius = (DEFAULT_ABS_SQUARE_WIDTH + 1) // 2
    TARGET_FOURIER_ORDER = 1
    FOURIER_AMPLITUDE = 0.1
    fourier_sample_absorption_image = np.where(
        np.logical_and(
            np.abs(y_indices - center_y_index) < box_radius,
            np.abs(x_indices - center_x_index) < box_radius
        ), 
        np.exp(-(1.0 - FOURIER_AMPLITUDE * np.cos(np.pi * FOURIER_SAMPLE_ORDER * ((y_indices - center_y_index) / box_radius)))),
        1.0
    )
    if not crop_to_roi:
        return fourier_sample_absorption_image 
    else:
        roi_xmin, roi_ymin, roi_xmax, roi_ymax = DEFAULT_ABSORPTION_IMAGE_CLOSE_ROI
        return fourier_sample_absorption_image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]

#Create a dummy measurement folder with a single (simulated) image, 
#plus experiment_parameters.json file and run_params_dump.json file.
#If the measurement is of a type that has multiple images per run, the same image stack is used for each.
def create_dummy_measurement_folder(image_stack, run_param_values, experiment_param_values, measurement_type):
    dummy_measurement_folder_pathname = os.path.join(TEMP_WORKSPACE_PATH, TEMP_MEASUREMENT_FOLDER_NAME)
    if not os.path.exists(dummy_measurement_folder_pathname):
        os.mkdir(dummy_measurement_folder_pathname)
    else:
        shutil.rmtree(dummy_measurement_folder_pathname) 
        os.mkdir(dummy_measurement_folder_pathname)
    #Create fake images
    DUMMY_RUN_ID = 1
    run_names = measurement.MEASUREMENT_IMAGE_NAME_DICT[measurement_type]
    for run_name in run_names:
        dummy_run_image_filename = "{0:d}_{1}_{2}.fits".format(DUMMY_RUN_ID, EPOCH_TIMESTRING_FILENAME, run_name)
        dummy_run_image_pathname = os.path.join(dummy_measurement_folder_pathname, dummy_run_image_filename)
        save_run_image(image_stack, dummy_run_image_pathname)
    #Create fake experiment parameters json
    update_time_dict = {} 
    for key in experiment_param_values:
        update_time_dict[key] = EPOCH_TIMESTRING_FILENAME
    experiment_parameters_dict = {}
    experiment_parameters_dict["Values"] = experiment_param_values 
    experiment_parameters_dict["Update_Times"] = update_time_dict
    experiment_parameters_pathname = os.path.join(dummy_measurement_folder_pathname, measurement.Measurement.MEASUREMENT_FOLDER_EXPERIMENT_PARAMS_FILENAME)
    with open(experiment_parameters_pathname, 'w') as f:
        json.dump(experiment_parameters_dict, f)
    #Create fake run parameters json
    if not "id" in run_param_values:
        run_param_values["id"] = DUMMY_RUN_ID 
    if not "runtime" in run_param_values:
        run_param_values["runtime"] = EPOCH_TIMESTRING_PARAMETERS
    run_parameters_dict = {}
    run_parameters_dict[str(DUMMY_RUN_ID)] = run_param_values 
    run_parameters_pathname = os.path.join(dummy_measurement_folder_pathname, measurement.Measurement.MEASUREMENT_FOLDER_RUN_PARAMS_FILENAME)
    with open(run_parameters_pathname, 'w') as f:
        json.dump(run_parameters_dict, f)
    return dummy_measurement_folder_pathname
    
#Given an array of floats representing the absorption of an image (with - dark)/(without - dark), generate 
#a dummy image stack to replicate the absorption image.

def generate_image_stack_from_absorption(absorption_array):
    abs_array_shape = absorption_array.shape 
    dark_image = np.full(abs_array_shape, BASE_DARK_LEVEL, dtype = np.ushort)
    without_image = np.full(abs_array_shape, BASE_LIGHT_LEVEL + BASE_DARK_LEVEL, dtype = np.ushort)
    with_image_unrounded = np.ones(abs_array_shape) * BASE_DARK_LEVEL + np.ones(abs_array_shape) * BASE_LIGHT_LEVEL * absorption_array 
    with_image = np.round(with_image_unrounded).astype(np.ushort)
    return np.stack((with_image, without_image, dark_image))


def save_run_image(image_stack, pathname):
    hdu = astropy.io.fits.PrimaryHDU(image_stack)
    hdu.writeto(pathname)



