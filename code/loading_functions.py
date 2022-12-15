import importlib.resources as pkg_resources
import json
import os 
import subprocess
import sys


def load_satyendra():
    path_to_file = os.path.dirname(os.path.abspath(__file__))
    path_to_satyendra = path_to_file + "/../../"
    sys.path.insert(0, path_to_satyendra)


def load_run_parameters_from_json(parameters_path, make_raw_parameters_terse = False):
    with open(parameters_path, 'r') as json_file:
        parameters_dict = json.load(json_file)
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



def load_experiment_parameters_from_folder(folder_path):
    parameters_path = os.path.join(folder_path, "experiment_parameters.json")
    with open(parameters_path, 'r') as json_file:
        return json.load(json_file)

def universal_clipboard_copy(text_to_copy):
    if(sys.platform.startswith("darwin")):
        #Copy command for MacOS 
        subprocess.run("pbcopy", universal_newlines = True, input = text_to_copy)
    elif(sys.platform.startswith("win32")):
        #Copy command for Windows 
        subprocess.run("clip", universal_newlines= True, input = text_to_copy)
    else:
        raise RuntimeError("Unsupported operating system: " + sys.platform)