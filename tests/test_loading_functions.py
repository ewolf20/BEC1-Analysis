import hashlib
import json
import os 
import sys 

import numpy as np
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


def test_update_nested_json():
    temp_filename = "temp_json.json"
    temp_pathname = os.path.join("resources", temp_filename) 
    try:
        with open(temp_pathname, 'w') as json_file:
            json.dump({}, json_file)
        loading_functions.update_nested_json(temp_pathname, 'foo', 1)
        try:
            loading_functions.update_nested_json(temp_pathname, 'bar', 2, parent_keys = ['a', 'b'], create_new = False)
        except KeyError:
            pass 
        else:
            assert False
        loading_functions.update_nested_json(temp_pathname, 'bar', 2, parent_keys = ['a', 'b'], create_new = True)
        with open(temp_pathname, 'r') as json_file:
            updated_dict = json.load(json_file)
        assert len(updated_dict) == 2
        assert updated_dict['foo'] == 1
        assert 'a' in updated_dict 
        assert 'b' in updated_dict['a']
        assert updated_dict['a']['b']['bar'] == 2
    finally:
        os.remove(temp_pathname)



def test_load_unitary_EOS():
    loaded_eos_dict = loading_functions.load_unitary_EOS()
    for key in loaded_eos_dict:
        assert isinstance(loaded_eos_dict[key], np.ndarray)
    listified_eos_dict = {key:list(loaded_eos_dict[key]) for key in loaded_eos_dict}
    dump_string = json.dumps(listified_eos_dict)
    dump_string_bytes = dump_string.encode("ASCII")
    dump_string_hash = get_sha_hash_string(dump_string_bytes)
    EXPECTED_DUMP_STRING_HASH = "094000a7f61ffebe97dd14ede5167cf4d50079a27b6775c9eefe2f0cd5261c1d"
    assert dump_string_hash == EXPECTED_DUMP_STRING_HASH