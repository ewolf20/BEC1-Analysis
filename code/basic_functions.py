import numpy as np





"""
Returns an absorption image, i.e. the ratio of the light counts at a given pixel with and without 
atoms, corrected by the dark image counts. If norm_box_coordinates is specified, uses those coordinates
to normalize the image_with_atoms and the image_without_atoms to have the same counts there. """
def get_absorption_image(image_stack, ROI = None, norm_box_coordinates = None, clean_image = True, clean_strategy = "default"):
    image_with_atoms, image_without_atoms, image_dark = image_stack
    if(norm_box_coordinates):
        norm_x_min, norm_y_min, norm_x_max, norm_y_max = norm_box_coordinates
        norm_with_atoms = image_with_atoms[norm_y_min:norm_y_max, norm_x_min:norm_x_max]
        norm_without_atoms = image_without_atoms[norm_y_min:norm_y_max, norm_x_min:norm_x_max] 
        norm_dark = image_dark[norm_y_min:norm_y_max, norm_x_min:norm_x_max]
        with_atoms_light_sum = sum(sum(norm_with_atoms - norm_dark))
        without_atoms_light_sum = sum(sum(norm_without_atoms - norm_dark)) 
        with_without_light_ratio = with_atoms_light_sum / without_atoms_light_sum 
    else:
        with_without_light_ratio = 1
    if(ROI):
        roi_x_min, roi_y_min, roi_x_max, roi_y_max = ROI
        image_with_atoms_ROI = image_with_atoms[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        image_without_atoms_ROI = with_without_light_ratio * image_without_atoms[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        image_dark_ROI = image_dark[roi_y_min:roi_y_max, roi_x_min:roi_y_max]
        absorption_image = (image_with_atoms_ROI - image_dark_ROI) / (image_without_atoms_ROI - image_dark_ROI) 
    else:
        image_without_atoms = image_without_atoms * with_without_light_ratio 
        absorption_image = (image_with_atoms - image_dark) / (image_without_atoms - image_dark) 
    if(clean_image):
        absorption_image = _clean_absorption_image(absorption_image, strategy = clean_strategy)
    return absorption_image

"""Cleans an absorption image.

strategies:
'default': Uses numpy's nan_to_num, which changes np.nan to 0 and np.inf to a very large number.'"""
def _clean_absorption_image(abs_image, strategy = 'default'):
    if(strategy == "default"):
        return np.nan_to_num(abs_image)


"""
Perform a naive pixel sum over an image.

ROI: An iterable [x_min, y_min, x_max, y_max] of the min and max x and y-coordinates. If none, defaults 
to summing over entire image."""
def pixel_sum(image, ROI = None):
    if(ROI):
        x_min, y_min, x_max, y_max = ROI 
        cropped_image = image[y_min:y_max, x_min:x_max]
        return sum(sum(cropped_image)) 
    else:
        return sum(sum(image))







