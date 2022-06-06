import hashlib
import importlib.resources as pkg_resources
import json
import os 
import sys 

path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"
sys.path.insert(0, path_to_analysis)

TEST_DIRECTORY_PATH = "./resources"

from BEC1_Analysis.code import crypto_functions 
from BEC1_Analysis import secrets as s




def check_sha_hash(my_bytes, checksum_string):
    m = hashlib.sha256() 
    m.update(my_bytes) 
    return m.hexdigest() == checksum_string

def get_sha_hash_string(my_bytes):
    m = hashlib.sha256() 
    m.update(my_bytes) 
    return m.hexdigest()



def test_initialize_fernet():
    f = crypto_functions.initialize_fernet()
    with open(os.path.join(TEST_DIRECTORY_PATH, 'test_encrypted_hello.bin'), 'rb') as encrypted_hello_file:
        encryption_bytes = encrypted_hello_file.read()
    assert f.decrypt(encryption_bytes) == b'hello'


def test_update_plaintext_experiment_parameters():
    ENCRYPTED_TEMP_FILENAME = 'encrypted_params_test.bin'
    PLAINTEXT_TEMP_FILENAME = 'plaintext_params_test.json'
    try:
        f = crypto_functions.initialize_fernet() 
        test_dict = {'foo':1, 'bar':2}
        test_dumps = json.dumps(test_dict) 
        test_encrypted = f.encrypt(test_dumps.encode('ASCII'))
        with pkg_resources.path(s, '__init__.py') as secrets_init_path:
            secrets_path = os.path.dirname(secrets_init_path) 
        with open(os.path.join(secrets_path, ENCRYPTED_TEMP_FILENAME), 'wb') as test_encrypted_file:
            test_encrypted_file.write(test_encrypted) 
        crypto_functions.update_plaintext_experiment_parameters(plaintext_filename = PLAINTEXT_TEMP_FILENAME,
                                                                encrypted_filename = ENCRYPTED_TEMP_FILENAME)
        with open(os.path.join(secrets_path, PLAINTEXT_TEMP_FILENAME)) as test_plaintext_file:
            retrieved_dict = json.load(test_plaintext_file)
        assert retrieved_dict == test_dict
    finally:
        os.remove(os.path.join(secrets_path, ENCRYPTED_TEMP_FILENAME))
        os.remove(os.path.join(secrets_path, PLAINTEXT_TEMP_FILENAME))


def test_update_encrypted_experiment_parameters():
    ENCRYPTED_TEMP_FILENAME = 'encrypted_params_test.bin'
    PLAINTEXT_TEMP_FILENAME = 'plaintext_params_test.json'
    try:
        test_dict = {'foo':1, 'bar':2}
        with pkg_resources.path(s, '__init__.py') as secrets_init_path:
            secrets_path = os.path.dirname(secrets_init_path) 
        with open(os.path.join(secrets_path, PLAINTEXT_TEMP_FILENAME), 'w') as test_plaintext_file:
            json.dump(test_dict, test_plaintext_file)
        crypto_functions.update_encrypted_experiment_parameters(plaintext_filename = PLAINTEXT_TEMP_FILENAME,
                                                                encrypted_filename = ENCRYPTED_TEMP_FILENAME)
        f = crypto_functions.initialize_fernet() 
        with open(os.path.join(secrets_path, ENCRYPTED_TEMP_FILENAME), 'rb') as test_encrypted_file:
            retrieved_bytes = test_encrypted_file.read()
            retrieved_dump_string = f.decrypt(retrieved_bytes).decode('ASCII')
            retrieved_dict = json.loads(retrieved_dump_string)
        assert retrieved_dict == test_dict
    finally:
        os.remove(os.path.join(secrets_path, ENCRYPTED_TEMP_FILENAME))
        os.remove(os.path.join(secrets_path, PLAINTEXT_TEMP_FILENAME))