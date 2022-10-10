import importlib.resources as pkg_resources
import json
import os 
import sys



def load_experiment_parameters():
    from .. import secrets as s 
    with pkg_resources.path(s, "experiment_parameters_secret.json") as parameters_path:
        with open(parameters_path) as parameters_file:
            return json.load(parameters_file)

def load_satyendra():
    path_to_file = os.path.dirname(os.path.abspath(__file__))
    path_to_satyendra = path_to_file + "/../../"
    sys.path.insert(0, path_to_satyendra)