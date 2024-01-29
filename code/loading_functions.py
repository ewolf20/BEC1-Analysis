import importlib.resources as pkg_resources
import json
import os 
import subprocess
import sys
import time

import numpy as np

from .. import resources as r

def load_run_parameters_from_json(parameters_path, make_raw_parameters_terse = False, have_patience = False):
    PATIENT_RETRIES = 10 
    PATIENT_SLEEP_TIME = 1.0
    IMPATIENT_RETRIES = 3
    IMPATIENT_SLEEP_TIME = 0.3
    if have_patience:
        load_retries = PATIENT_RETRIES
        load_sleep_time = PATIENT_SLEEP_TIME
    else:
        load_retries = IMPATIENT_RETRIES 
        load_sleep_time = IMPATIENT_SLEEP_TIME
    parameters_dict = _load_json_with_retries(parameters_path, retries = load_retries, sleep_time = load_sleep_time)
    unsorted_parameters_list = [] 
    for key in parameters_dict:
        run_id = int(key)
        run_parameters = parameters_dict[key]
        if make_raw_parameters_terse:
            terse_run_parameters = {} 
            terse_run_parameters['id'] = run_parameters['id'] 
            terse_run_parameters['runtime'] = run_parameters['runtime']
            list_bound_variable_names = run_parameters["ListBoundVariables"]
            for list_bound_variable_name in list_bound_variable_names:
                terse_run_parameters[list_bound_variable_name] = run_parameters[list_bound_variable_name] 
            if "badshot" in run_parameters:
                terse_run_parameters["badshot"] = run_parameters["badshot"]
            unsorted_parameters_list.append((run_id, terse_run_parameters))
        else:
            unsorted_parameters_list.append((run_id, run_parameters))
    sorted_parameters_list = [f[1] for f in sorted(unsorted_parameters_list, key = lambda x: x[0])] 
    return sorted_parameters_list


#Hack method to load json files that may be being written to by other 
#threads/processes
def _load_json_with_retries(pathname, retries = 0, sleep_time = 0.0):
    counter = 0
    while True:
        try:
            with open(pathname, 'r') as json_file:
                return json.load(json_file)
        except (json.JSONDecodeError, OSError, FileNotFoundError) as e:
            print("Got error")
            counter += 1
            if counter < retries:
                time.sleep(sleep_time)
            else:
                raise e

def load_experiment_parameters_from_folder(folder_path):
    parameters_path = os.path.join(folder_path, "experiment_parameters.json")
    with open(parameters_path, 'r') as json_file:
        return json.load(json_file)



"""
Convenience method for manipulating nested JSON files. 

Given a json file located at pathname, adds a key-value pair to the file. If parent_keys is specified, 
navigates down through a hierarchy of dicts with respective keys in parent keys before adding the key-value pair.

Parameters:

create_new: If True, the method will create new dicts under the keys in parent keys if they are not present.

parent_keys: Iterable [key1, key2, ...] of keys. If passed, the key-value pair is added into the json file as follows:
{
    key_1: {
        key_2:{
            key:value
        }
    }
}"""
def update_nested_json(pathname, key, value, parent_keys = None, create_new = False):
    if parent_keys is None:
        parent_keys = []
    with open(pathname, 'r') as json_file:
        pre_existing_dict = json.load(json_file)
    modified_dict = pre_existing_dict
    for parent_key in parent_keys:
        if parent_key in modified_dict:
            modified_dict = modified_dict[parent_key]
        elif create_new:
            modified_dict[parent_key] = {} 
            modified_dict = modified_dict[parent_key]
        else:
            raise KeyError("Parent key not in json")
    modified_dict[key] = value
    with open(pathname, 'w') as json_file:
        json.dump(pre_existing_dict, json_file)


def load_unitary_EOS():
    UNITARY_EOS_FILENAME = "Experimental_Unitary_Fermi_Gas_EOS.csv"
    with pkg_resources.path(r, UNITARY_EOS_FILENAME) as eos_path:
        with open(eos_path, 'r') as eos_file:
            eos_data = np.loadtxt(eos_file, delimiter =",", skiprows = 1, unpack = True)
            (kappa_tilde, p_tilde, Cv_over_Nk, T_over_TF, E_over_E0, mu_over_EF, 
            F_over_E0, S_over_NkB, betamu) = eos_data
            eos_dict = {
                "kappa_over_kappa0":kappa_tilde, 
                "P_over_P0":p_tilde, 
                "Cv_over_NkB":Cv_over_Nk, 
                "T_over_TF":T_over_TF, 
                "E_over_E0":E_over_E0, 
                "mu_over_EF":mu_over_EF, 
                "F_over_E0":F_over_E0, 
                "S_over_NkB":S_over_NkB, 
                "betamu":betamu
            }
            return eos_dict

def load_polylog_analytic_continuation_parameters():
    CENTERS_FILENAME = "Polylog_Taylor_Centers.npy"
    COEFFS_3_2_FILENAME = "Polylog_Taylor_Coefficients_3_2.npy"
    COEFFS_5_2_FILENAME = "Polylog_Taylor_Coefficients_5_2.npy"
    with pkg_resources.path(r, CENTERS_FILENAME) as centers_path, \
        pkg_resources.path(r, COEFFS_3_2_FILENAME) as coeffs_3_2_path, \
        pkg_resources.path(r, COEFFS_5_2_FILENAME) as coeffs_5_2_path:
        centers = np.load(centers_path)
        coeffs_3_2 = np.load(coeffs_3_2_path) 
        coeffs_5_2 = np.load(coeffs_5_2_path) 
    return (centers, coeffs_3_2, coeffs_5_2)


def load_tabulated_ideal_betamu_vs_T_over_TF():
    TABULATED_BETAMU_FILENAME = "Tabulated_Ideal_Betamu_vs_T_over_TF.npy"
    with pkg_resources.path(r, TABULATED_BETAMU_FILENAME) as betamu_path:
        return np.load(betamu_path)

def universal_clipboard_copy(text_to_copy):
    if(sys.platform.startswith("darwin")):
        #Copy command for MacOS 
        subprocess.run("pbcopy", universal_newlines = True, input = text_to_copy)
    elif(sys.platform.startswith("win32")):
        #Copy command for Windows 
        subprocess.run("clip", universal_newlines= True, input = text_to_copy)
    else:
        raise RuntimeError("Unsupported operating system: " + sys.platform)