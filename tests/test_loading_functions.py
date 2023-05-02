import hashlib
import json
import os 
import sys 

import matplotlib.pyplot as plt 

path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)

from BEC1_Analysis.code import loading_functions

def get_sha_hash_string(my_bytes):
    m = hashlib.sha256() 
    m.update(my_bytes) 
    return m.hexdigest()

def test_load_experiment_parameters_from_folder():
    PARAMETERS_FOLDER_PATH = "resources/Test_Measurement_Directory"
    EXPECTED_DICT = {"Values":{}, "Update_Times":{}}
    experiment_parameters_dict = loading_functions.load_experiment_parameters_from_folder(PARAMETERS_FOLDER_PATH)
    assert (experiment_parameters_dict == EXPECTED_DICT)


def test_load_run_parameters_from_json():
    RUN_PARAMETERS_PATH = "resources/Test_Measurement_Directory/run_params_dump.json"
    RUN_PARAMS_VERBOSE_SHA_CHECKSUM = "8aafd39c67bc75e9736dd5822a9760abb748f7f69d17e7300a0c33d66360a2fc"
    RUN_PARAMS_TERSE_SHA_CHECKSUM = "a53edcdea468ed87165fffb8757742b1ddd9fed57aee0754b572bd37ee3369d8" 
    run_parameters_verbose = loading_functions.load_run_parameters_from_json(RUN_PARAMETERS_PATH, make_raw_parameters_terse = False)
    run_parameters_verbose_checksum = get_sha_hash_string(json.dumps(run_parameters_verbose).encode("ASCII"))
    assert run_parameters_verbose_checksum == RUN_PARAMS_VERBOSE_SHA_CHECKSUM
    run_parameters_terse = loading_functions.load_run_parameters_from_json(RUN_PARAMETERS_PATH, make_raw_parameters_terse = True)
    run_parameters_terse_checksum = get_sha_hash_string(json.dumps(run_parameters_terse).encode("ASCII"))
    assert run_parameters_terse_checksum == RUN_PARAMS_TERSE_SHA_CHECKSUM


