import numpy as np 


from . import data_fitting_functions, image_processing_functions



"""
A badshot function which detects bad shots by looking at absorption image pixel sums in the norm box and flagging as bad shots 
those which differ from the median by too much. Intended to catch andor glitches.
"""

def norm_box_filter(runs_dict, **kwargs):
    MULTIPLICATIVE_LEEWAY = 3.0
    counts_dict = {}
    for run_id in runs_dict:
        current_run = runs_dict[run_id]
        if not current_run.is_badshot:
            current_image_stack = current_run.get_default_image()
            current_abs_image = image_processing_functions.get_absorption_image(current_image_stack, ROI = kwargs['norm box'])
            counts_dict[run_id] = image_processing_functions.pixel_sum(current_abs_image)
    counts_list_sorted = sorted([counts_dict[id] for id in counts_dict])
    median_counts = counts_list_sorted[len(counts_list_sorted) // 2]
    badshots_list = [] 
    for run_id in counts_dict:
        current_counts = counts_dict[run_id] 
        if current_counts > MULTIPLICATIVE_LEEWAY * median_counts or current_counts < median_counts / MULTIPLICATIVE_LEEWAY:
            badshots_list.append(run_id) 
    return badshots_list