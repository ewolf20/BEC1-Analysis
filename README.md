# BEC1-Analysis
Analysis code for the Zwierlein BEC1 lab

# Purpose
This code has been written with the primary goal of streamlining, unifying, and speeding up data analysis in Prof. Martin Zwierlein's BEC1 laboratory at the MIT Research Laboratory for Electronics (https://www.rle.mit.edu/quantumgas/portfolio/bec1/). To this end, it implements a standard framework to hold raw data from the experiment, along with a wide variety of functions relevant to its analysis.

Some of these functions are highly specific to the BEC1 experiment, but others - e.g. fitting functions or basic image processing functions - could be relevant to other quantum gas experiments. As such, an effort has been made to make the code maximally adaptable to use in other quantum gas many-body experiments whenever this doesn't interfere with the above.

Note that things related to the **acquisition** of data - interfacing with cameras, controlling analog output cards, etc. - are outside of the scope of this code. 

# Installation 

At present, BEC1_Analysis is implemented with a module structure, but not as a package. To use it, you must manually pull it from Github at the following 
address: https://github.com/ewolf20/BEC1-Analysis.

Additionally, because BEC1_Analysis is a module, not a package, you must manually install its dependencies. Navigate to the root folder of BEC1_Analysis, then run 

    pip install -r requirements.txt

Finally, there is one extra user-level build step: when you pull the source of BEC1_Analysis to a new computer, you should navigate to 

    BEC1_Analysis/code/c_code

and call 

    python build_polrot_code.py


Finally, if you wish to test whether the code is fully functional, you may navigate to the BEC1_Analysis/tests and call 

    pytest 

Any failures indicate that something is wrong.

## Importing BEC1_Analysis 

At present, BEC1_Analysis is written as a Python module, but not a package. What this means is that while you can write e.g. 

    from BEC1_Analysis.code import image_processing_functions 
    from BEC1_Analysis.tests import test_image_processing_functions 

you will first have to specify the location of BEC1_Analysis to Python before you import it. Supposing that your directory structure looks like e.g.

    C:/Users/Me/Repos/BEC1_Analysis

you would need to add the path C:/Users/Me/Repos to your PYTHONPATH or to sys.path directly. A convenient formula to do this in a script is as follows:

    import sys
    path_to_folder = "C:/Users/Me/Repos/BEC1_Analysis"
    sys.path.insert(0, path_to_folder)

    #Import whatever you need 
    from BEC1_Analysis.code import ...


# Description and Quickstart

## Run and Measurement Classes

The basic core of this package is the Measurement class. This class wraps a dict of Run objects, which correspond to the output of the machine for a single iteration of the experimental sequence (image files, programmatically-set variables, etc.), handling the task of analyzing images run-by-run behind the scenes. The Measurement class also has information about stable experiment parameters (e.g. trapping frequencies), plus user-configured parameters for analysis.

To begin, one should create a measurement as follows:


    from BEC1_Analysis.code.measurement import Measurement
    my_measurement = Measurement("/name_of_your_measurement_directory", imaging_mode = "desired_imaging_mode")

The measurement object assumes that this directory contains the following things:

- Run image files 

- Run Parameters

- Experiment Parameters

For more detail on the structure of the measurement directory, see the section "Code Structure: Deeper Dive", below. 

## Setting ROI and Norm Boxes

The vast majority of analysis functions will require a region of interest (ROI) as well as a normalization box (norm_box) to be specified on the images being analyzed:

    my_measurement.set_ROI() 
    my_measurement.set_norm_box() 

Other boxes can be specified:

    my_measurement.set_box("secret_box")

Invoking these methods without further arguments will open up a graphical dialog for seting the boxes; simply click and drag a rectangle over the region of interest, then close the figure when you are happy. By default, the first image in the sequence is used for this; if it is unsuitable, you may close the figure without selecting a rectangle to proceed to the next image instead.

It is also supported to directly pass the coordinates of a box as a list:

    \# Syntax is [xmin, ymin, xmax, ymax]
    ROI = [100, 200, 150, 240]
    my_measurement.set_ROI(box_coordinates = ROI)

## Applying Analyses

It is possible to access all of the raw information in a Measurement object directly; one can pull runs from the underlying runs_dict object, call their get_image methods, access their parameter dictionaries, etc. However, analysis can be streamlined by applying an analysis function to all runs, then accessing the results directly:

    my_measurement.analyze_runs(my_analysis_func, "foo") 
    vals = my_values.get_analysis_value_from_runs("foo") 

For convenience, returned values are numpy arrays by default:

    vals_div_by_2 = vals / 2.0

The above methods support exception handling, the ability to ignore flagged "bad shots", arbitrary filtering of the runs to which the analysis is applied or from which the values are read out, and the ability to pass kwargs to the analysis function. See the documentation (work in progress) for details.

### The Analysis Function

Analysis functions should be written with the signature 

    my_analysis_func(my_measurement, my_run, kwarg_1 = val1, ...). 
    
Such functions may be used to wrap all of the low-level interfacing with the Measurement and Run objects - for example, calls to my_run.get_image() or querying of my_run.parameters[value_name]. Analysis functions may return arbitrary datatypes. There is special syntax for analysis functions which, by their nature, return two or more distinct quantities of interest:

    def my_analysis_func(my_measurement, my_run):
        temperature = calculate_temp(my_measurement, my_run) 
        energy = calculate_energy(my_measurement, my_run)
        return (temperature, energy) 

    my_measurement.analyze_runs(my_analysis_func, ("temperature", "energy"))
    my_temperatures = my_measurement.get_analysis_value_from_runs("temperature") 
    my_energies = my_measurement.get_analysis_value_from_runs("energy")


Examples analysis functions which are relevant to BEC1 may be found in the source code at code/analysis_functions.py

### The Badshot function 

Especially relevant to live analyses (see below), the badshot function is essentially a special analysis function - it has identical signature - which is assumed to return True or False depending on whether a given run is or is not a bad shot. Runs which are flagged as bad shots are, by default, excluded from processing when analyses are applied. 

Runs may be flagged as bad shots in a few ways.

- Manually, by passing run_ids to flag as bad shots:

        my_measurement.label_badshots_custom(badshot_list = [bad_run_id_1, bad_run_id_2, ...])

- Algorithmically, by passing a custom function which flags runs as bad shots:

        my_badshot_function(my_measurement, my_run):
            return (my_run.parameters["id"] in [bad_run_id_1, bad_run_id_2, ...])

        my_measurement.label_badshots_custom(badshot_function = my_badshot_function)

- Algorithmically and repeatably, by setting a default bad shot function:

        my_badshot_function(my_measurement, my_run):
            return (my_run.parameters["id"] % 2) != 0

        my_measurement.set_badshot_function(my_badshot_function)
        my_measurement._label_badshots_default()
        \# or this 
        my_measurement.update()


## Accessing Analysis Results

The default syntax for accessing analysis results is straightforward:

    my_vals = my_measurement.get_analysis_value_from_runs("val") 

By default, the list of values for each run is converted to a numpy array for convenience:

    my_vals = my_measurement.get_analysis_value_from_runs("val") 
    /# This works as long as val is a number
    my_divided_vals = my_vals / 3.0

However, this can be disabled:

    my_vals = my_measurement.get_analysis_value_from_runs("val", numpyfy = False)
    \# Now the result is a list 
    for val in my_vals:
        \# Do something

There is a similar syntax for accessing parameter values from runs:

    my_ids = my_measurement.get_parameter_value_from_runs("id") 

and even a method for accessing both at once (useful when, e.g., errors are being filtered):

    my_ids, my_vals = my_measurement.get_parameter_analysis_value_pair_from_runs("id", "val")


As with applying analyses, it is possible to ignore bad shots, filter out errors, and apply arbitrary filters to the runs which return here; see documentation.

## Live Analysis

It is often useful to be able to monitor certain basic analysis results in real time - for example, atom counts vs. AOM frequency when trying to find a resonance. This could be accomplished by repeated re-checking of the measurement directory for new runs and subsequent calls to analyze_runs:

    while True:
        my_measurement._update_runs_dict() 
        my_measurement.analyze_runs(my_analysis_func, "foo")
        vals = my_measurement.get_analysis_value_from_runs("foo")
        use_vals(vals)
        time.sleep(SLEEP_TIME)

However, for live analyses, it would typically be best to only analyze new runs, and to catch errors so the whole code doesn't crash; one may also wish to filter out bad shots in real time, again only applying the filter to new runs:

    while True:
        my_measurement._update_runs_dict() 
        my_measurement.filter_badshots_custom(badshot_function = my_badshot_function, 
                                        overwrite_existing_badshots = False, use_badshots_checked_list = True)
        my_measurement.analyze_runs(my_analysis_func, "foo", overwrite_existing = False, catch_errors = True, ignore_badshots = True)
        vals = my_measurement.get_analysis_value_from_runs("foo", ignore_errors = True, ignore_badshots = True)
        use_vals(vals)
        time.sleep(SLEEP_TIME)

To make this process more transparent, an update() function has been added for convenience; the above is equivalent to 

    my_measurement.add_to_live_analyses(my_analysis_func, "foo")
    my_measurement.set_badshot_function(my_badshot_function)
    while True:
        my_measurement.update() 
        vals = my_measuremet.get_analysis_value_from_runs("foo", ignore_errors = True, ignore_badshots = True)
        use_vals(vals) 
        time.sleep(SLEEP_TIME) 

Note that analyses are applied in the order that they are added:

    def my_analysis_func(my_measurement, my_run):
        return 0.0 
    def my_other_analysis_func(my_measurement, my_run):
        return my_run.analysis_results["foo"] + 1.0

    my_measurement.add_to_live_analyses(my_analysis_func, "foo") 
    my_measurement.add_to_live_analyses(my_analysis_func, "bar") 
    my_measurement.update()
    #Runs fine


## Analysis dumps

For convenience, it has been made possible to "dump" the entire contents of the analysis results for a measurement, then re-load them at a later date to be re-examined without potentially lengthy recalculation:

    my_measurement.dump_analysis(pathname) 
    my_measurement.load_analysis(pathname)

JSON is used to serialize values for this, so this functionality is not supported (and will throw an error) for analysis results which are not JSON-serializable. The single exception is numpy arrays, which are handled via a workaround. 


# Code Structure: Deeper Dive

## Measurement Directory Structure 

As stated above, the directory from which a measurement is created must contain run image files, run parameters, and experiment parameters.

### Run image files

In quantum gas experiments, a very standard paradigm is to image a collection of atoms in three "shots" - one with atoms, one with light and without atoms, and one with neither light nor atoms - in order to determine optical density/fluorescence intensity while correcting for the effects of stray light or dark counts. This code assumes that every image associated with a run corresponds to a file containing these three "shots" as arrays of photon counts on the pixel array of some fast camera. At present, a .fits image is assumed. 

It is further assumed that run image files are named according to the following syntax: 
>RunID_DateCreated_ImageName.extension

where 

- Run id is an integer which is unique to that experimental run, ideally globally but at least within the measurement directory.
- DateCreated is a datetime when the run was taken, formatted according to Measurement.DATETIME_FORMAT_STRING
- ImageName is a label for the specific image, used because in some cases runs have multiple images (though even single-image runs have this label)

Note that these image files are generically very large; as such, by default the measurement object does not load them into memory when initialized, but instead loads them "on demand" when requested/needed for analysis. For small datasets, a performance gain can likely be realized by setting hold_images_in_memory to True in the measurement kwargs.

### Run Parameters

Generically, quantum gas experiments will have a host of parameters which can be set by control software - for example, AOM frequencies or imaging delay times. Typical datasets consist of varying these parameters and looking for changes in the atomic cloud. As such, each run must be associated with its parameters for analysis. This is done via a .json file, named as given by the constant Measurement.RUN_PARAMS_FILENAME. This json file is assumed to be a dict of {RunID:parameters_dictionary} value pairs, where RunID is the same integer specified in the run image filenames and parameters_dictionary is an arbitrary dictionary of key:value pairs. 

### Experiment parameters

In contrast to software-controlled parameters, most experiments will have some parameters which are dictated by the setup and must be measured - imaging magnification, say, or magnetic trap frequencies. These parameters generally do not vary within an experimental dataset, but are still essential for its analysis. Accordingly, it is assumed that there is a .json file, named as given by the constant Measurement.EXPERIMENT_PARAMS_FILENAME, containing these parameters as key:value pairs. 


## Measurement and Run Class Structure 

For the end-user who wishes to write custom analysis functions, it is necessary to understand a little more about the structure of measurements and runs. 

### Run Structure

A Run encodes data about a single cycle of the experiment: one iteration of the atom shutter opening, the MOT being loaded, and images being taken. This data includes:

- Images. These are images from the experiment, assumed (as stated above) to be a three-item stack of two-dimensional images (with_atoms, without_atoms, without_light). These stacks are accessed by calling the get_image method of a run:

        image_stack = my_run.get_image(image_name)
        image_with_atoms, image_without_atoms, image_without_light = image_stack 
        \# The images are returned as 2d numpy arrays 
        image_with_atoms / image_without_light

    image_name is a string which labels the particular image from the run; it is the same string as appears in the run filename. 

    If images are being loaded on the fly, it is often advisable to enable memory-mapping, so that unused parts of the image aren't loaded into memory:

        image_stack = my_run.get_image(image_name, memmap = True)

- Parameters. These are values programmed manually into the control computer prior to a run. These are stored in a dictionary at the run level:

        imaging_frequency = my_run.parameters["imaging_frequency"]

    my_run.parameters is precisely the dictionary which is stored under the run's id in the run parameters JSON file described above. Please note that it is bad practice to modify this dictionary at runtime, though I can't stop you. 

- Analysis results. These are values obtained by analyses performed on the run, indexed by the label specified in analyze_runs:

        def my_analysis_func(my_measurement, my_run):
            return "Cheese is delicious at: " + my_run.parameters["runtime"]

        my_measurement.analyze_runs(my_analysis_func, "cheese_time")
        /# Get a run from the measurement by accessing by id
        my_run = my_measurement.runs_dict[1337] 
        cheese_string = my_run.analysis_results["cheese_time"]

- Bad shot status. This is, as described above, a boolean label saying whether a run is a bad shot. Most operations on runs - performing analyses, getting values, etc. - ignore runs which are labeled as bad shots by default. 

        if my_run.is_badshot:
            print("Woe is me") 

        /# This is the same 
        if my_run.analysis_results["badshot"]:
            print("Woe, I say!")


- ID. This can be accessed either from the parameters or directly:

        if my_run.run_id == 777:
            print("My lucky number") 

        /# This is the same 

        if my_run.parameters["id"] == 776:
            print("Aww, so close!")


## Measurement Structure 

A Measurement encodes data on an entire set of runs which, together, constitute a logical unit: it could for example be a measurement of an imaging resonance frequency, where an AOM frequency is scanned and the opacity of an atom cloud is imaged. The only hard requirement is that all of the runs in this logical unit have the same image_names associated with them. 

### For analysis functions


A measurement contains the following attributes that users writing custom analysis functions would commonly access:


- Experiment Parameters: This is a dictionary of semi-stable parameters of the experiment, encoding roughly the state of the experiment when data was taken; things like magnifications or trapping frequencies are here. It is precisely the dictionary encoded by the experiment parameters JSON file.

        /# You need this to calculate atom numbers 
        pixel_area = np.square(my_measurement.experiment_parameters["pixel_length"])

- Measurement parameters: This is a dictionary of parameters specified by the user at runtime, either manually or via functions like set_box():

        my_measurement.set_box("my_box")
        /# This is how the box coordinates are accessed and used
        box_coordinates = my_measurement.measurement_parameters["my_box"] 
        /# It is not bad practice to do this
        my_measurement.measurement_parameters["cutoff"] = 3.14

- Measurement analysis results: This is a dictionary where the results of measurement-wide analyses go:

        def my_badshot_function(my_measurement, my_run):
            return my_run.analysis_results["counts"] < my_measurement.measurement_analysis_results["cutoff_counts"]

    It is rare that you would have to use this; it's mostly for dynamic badshot functions.


### Other

For those adapting or updating the code, it's also useful to know about the following attributes:

- Runs dict: A measurement wraps its constituent Runs in a dictionary called runs_dict, where they are keyed by run_id:

        /# This is the kind of loop that would appear in manual analysis code
        frequency_list = []
        for run_id in my_measurement.runs_dict:
            my_run = my_measurement.runs_dict[run_id] 
            frequency_list.append(my_run.parameters["frequency"])


Most of the other attributes are just bookkeeping for live analysis.

# Support and Contributing

I (Eric Wolf) am currently the primary developer of BEC1_Analysis; it is my intent to support the code for its primary purpose of in-lab data analysis until the conclusion of my PhD. I anticipate that most users of the code will be able to contact me in-person with issues; all others may submit issues on the GitHub page.

Should you wish to contribute, feel free to talk to me in-person or open a PR on GitHub, being sure to follow the style of the repository as best you can. Note that any changes which make this code materially less convenient for me and my labmates to use will be rejected - you can, however, always fork the repo if you want.

I make no promises as to support for the code after conclusion of my PhD (likely in 2026).

# External Sources and Attributions

Values for most physical constants are taken from the NIST Reference on Constants, Units, and Uncertainty in ~2022-2023: 
https://physics.nist.gov/cuu/Constants/index.html


A few values specific to Li6 are obtained from Michael Gehm's nice review:
https://jet.physics.ncsu.edu/techdocs/pdf/PropertiesOfLi.pdf


Statistical mechanics formulas have been taken, with occasionally substantial modification, from Mehran Kardar's textbook:
Kardar, "Statistical Physics of Particles". Any errors should be assumed to be mine, at least initially. 

Experimental values for the unitary Fermi gas EOS are obtained from research cited in:

Ku, Mark JH, Ariel T. Sommer, Lawrence W. Cheuk, and Martin W. Zwierlein. "Revealing the superfluid lambda transition in the universal thermodynamics of a unitary Fermi gas." Science 335, no. 6068 (2012): 563-567.

Available (paywalled) at
https://doi.org/10.1126/science.1214987

Also available at 
https://arxiv.org/abs/1110.3309