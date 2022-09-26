import os 
import sys 

import numpy as np 
import matplotlib.pyplot as plt

path_to_file = os.path.dirname(os.path.abspath(__file__))
path_to_repo_folder = os.path.abspath(path_to_file + "/../../")


sys.path.insert(0, path_to_repo_folder)

from BEC1_Analysis.code import image_processing_functions, data_fitting_functions
from imaging_resonance_processing import get_workfolder_path


def main():
    file_path = get_data_file_input() 
    frequencies, counts = np.load(file_path) 
    fit_results, inlier_indices = data_fitting_functions.fit_imaging_resonance_lorentzian(frequencies, counts, filter_outliers = True, 
                                                                                        report_inliers = True) 
    popt, pcov = fit_results
    inlier_frequencies = frequencies[inlier_indices]
    inlier_counts = counts[inlier_indices]
    overall_indices = np.arange(len(frequencies))
    outlier_indices = overall_indices[~np.isin(overall_indices, inlier_indices)] 
    outlier_frequencies = frequencies[outlier_indices] 
    outlier_counts = counts[outlier_indices]
    fit_report = data_fitting_functions.fit_report(data_fitting_functions.imaging_resonance_lorentzian, fit_results)
    frequency_plot_range = np.linspace(min(frequencies), max(frequencies), 100)
    fit_plot_values = data_fitting_functions.imaging_resonance_lorentzian(frequency_plot_range, *popt)
    title = os.path.basename(file_path).split('.')[0]
    with open(os.path.join(get_workfolder_path(), title + "_Fit_Report.txt"), 'w') as f:
        f.write(fit_report) 
    print(fit_report)
    plt.plot(inlier_frequencies, inlier_counts, 'o', label = "Data") 
    plt.plot(frequency_plot_range, fit_plot_values, label = "Fit to data") 
    plt.plot(outlier_frequencies, outlier_counts, 'rd', label = "Outliers")
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