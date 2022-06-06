import sys 
import os 

path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_analysis = path_to_file + "/../../"

sys.path.insert(0, path_to_analysis)

from BEC1_Analysis.code import crypto_functions 


def main():
    crypto_functions.update_plaintext_experiment_parameters() 


if __name__ == "__main__":
    main()