import os 
import sys 

import numpy as np 
import matplotlib.pyplot as plt

path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_repo_folder = os.path.abspath(path_to_file + "/../../")


sys.path.insert(0, path_to_repo_folder)

from BEC1_Analysis.code import image_processing_functions, data_fitting_functions
from imaging_resonance_processing import get_workfolder_path


UPPER_COUNTS_CUTOFF = np.inf
LOWER_COUNTS_CUTOFF = -np.inf

def main():
    file_path = get_data_file_input() 
    frequencies, counts = np.load(file_path) 
    frequencies_trimmed, counts_trimmed = zip(*[f for f in zip(frequencies, counts) if f[1] < UPPER_COUNTS_CUTOFF and f[1] > LOWER_COUNTS_CUTOFF])
    frequencies_trimmed = list(frequencies_trimmed)
    counts_trimmed = list(counts_trimmed) 
    fit_results = data_fitting_functions.fit_imaging_resonance_lorentzian(frequencies_trimmed, counts_trimmed) 
    popt, pcov = fit_results
    fit_report = data_fitting_functions.fit_report(data_fitting_functions.imaging_resonance_lorentzian, fit_results)
    fit_values = data_fitting_functions.imaging_resonance_lorentzian(frequencies_trimmed, *popt)
    frequencies_sorted, fit_values_sorted = zip(*sorted(zip(frequencies_trimmed, fit_values), key = lambda f: f[0]))
    title = os.path.basename(file_path).split('.')[0]
    with open(os.path.join(get_workfolder_path(), title + "_Fit_Report.txt"), 'w') as f:
        f.write(fit_report) 
    print(fit_report)
    plt.plot(frequencies_trimmed, counts_trimmed, 'o', label = "Data") 
    plt.plot(frequencies_sorted, fit_values_sorted, label = "Fit to data") 
    plt.xlabel("Actual Freq (MHz, arb offset)") 
    plt.ylabel("Counts (arb)")
    plt.legend() 
    plt.suptitle(title)
    plt.savefig(os.path.join(get_workfolder_path(), title + "_Graph.png"))
    plt.show()
    




def get_data_file_input():
    print("Please enter the name of or full absolute path to the .npy file containing the processed counts data.") 
    print("If path is not specified, the file will be searched for in today's directory in Private_BEC1_Analysis.")
    user_input = input()
    if not os.path.isfile(user_input):
        file_path = os.path.join(get_workfolder_path(), user_input)
    else:
        file_path = user_input
    return file_path 


if __name__ == "__main__":
    main()