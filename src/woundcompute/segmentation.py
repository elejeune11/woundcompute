import numpy as np
from scipy import ndimage
from skimage import measure, morphology
from skimage.filters import threshold_otsu, gabor
from skimage.measure import label, regionprops
from typing import List, Union


def apply_median_filter(array: np.ndarray, filter_size: int) -> np.ndarray:
    """Given an image array. Will return the median filter applied by scipy"""
    filtered_array = ndimage.median_filter(array, filter_size)
    return filtered_array


def apply_gaussian_filter(array: np.ndarray, filter_size: int) -> np.ndarray:
    """Given an image array. Will return the gaussian filter applied by scipy"""
    filtered_array = ndimage.gaussian_filter(array, filter_size)
    return filtered_array


def compute_otsu_thresh(array: np.ndarray) -> Union[float, int]:
    """Given an image array. Will return the otsu threshold applied by skimage."""
    thresh = threshold_otsu(array)
    return thresh


def apply_otsu_thresh(array: np.ndarray) -> np.ndarray:
    """Given an image array. Will return a boolean numpy array with an otsu threshold applied."""
    thresh = compute_otsu_thresh(array)
    thresh_img = array > thresh
    return thresh_img


def get_region_props(array: np.ndarray) -> List:
    """Given a binary image. Will return the list of region props."""
    label_image = label(array)
    region_props = regionprops(label_image)
    return region_props


def get_largest_regions(region_props: List, num_regions: int = 3) -> List:
    """Given a list of region properties. Will return a list of the num_regions largest regions.
    If there are fewer than num_regions regions, will return all regions."""
    area_list = []
    for region in region_props:
        area_list.append(region.area)
    ranked = np.argsort(area_list)[::-1]
    num_to_return = np.min([len(ranked), num_regions])
    regions_list = []
    for kk in range(0, num_to_return):
        idx = ranked[kk]
        regions_list.append(region_props[idx])
    return regions_list


def get_regions_not_touching_bounds(region_props: List, img_shape: tuple) -> List:
    """Given a list of region properties. Will return a list of all region properties not touching the edges of the domain."""
    new_regions = []
    for region in region_props:
        coords = region.coords
        if 0 in coords:
            continue
        if img_shape[0] - 1 in coords[:, 0]:
            continue
        if img_shape[1] - 1 in coords[:, 1]:
            continue
        new_regions.append(region)
    return new_regions


def get_roundest_regions(region_props: List, num_regions: int = 3) -> List:
    """Given a list of region properties. Will return the num_regions roundest regions.
    If there are fewer than num_regions regions, will return all regions.
    For eccentricity, 0 = circle, 1 = more elliptical"""
    eccentricity_list = []
    for region in region_props:
        eccentricity = region.eccentricity
        eccentricity_list.append(eccentricity)
    ranked = np.argsort(eccentricity_list)
    num_to_return = np.min([len(ranked), num_regions])
    regions_list = []
    for kk in range(0, num_to_return):
        idx = ranked[kk]
        regions_list.append(region_props[idx])
    return regions_list


def get_domain_center(array: np.ndarray) -> Union[int, float]:
    """Given an array. Will return center (ix_0, ix_1)"""
    center_0 = array.shape[0] / 2.0
    center_1 = array.shape[1] / 2.0
    return center_0, center_1


def compute_distance(
    a0: Union[int, float],
    a1: Union[int, float],
    b0: Union[int, float],
    b1: Union[int, float]
) -> Union[int, float]:
    """Given two points. Will return distance between them."""
    dist = ((a0 - b0)**2.0 + (a1 - b1)**2.0)**0.5
    return dist


def get_closest_region(
    regions_list: List,
    loc_0: Union[int, float],
    loc_1: Union[int, float]
) -> object:
    """Given a list of region properties. Will return the object closest to location."""
    center_dist = []
    for region in regions_list:
        centroid = region.centroid
        region_0 = centroid[0]
        region_1 = centroid[1]
        dist = compute_distance(region_0, region_1, loc_0, loc_1)
        center_dist.append(dist)
    ix = np.argmin(center_dist)
    return regions_list[ix]


def extract_region_props(region_props: object) -> Union[float, np.ndarray]:
    """Given region properties from skimage.measure.regionprops.
    Will return the values of relevant properties.
    See: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
    """
    if region_props is None:
        return None, None, None, None, None, None, (None, None, None, None), None
    else:
        area = region_props.area
        axis_major_length = region_props.axis_major_length
        axis_minor_length = region_props.axis_minor_length
        centroid = region_props.centroid
        centroid_row = centroid[0]
        centroid_col = centroid[1]
        coords = region_props.coords
        bbox = region_props.bbox
        orientation = region_props.orientation
        return area, axis_major_length, axis_minor_length, centroid_row, centroid_col, coords, bbox, orientation


def region_to_coords(regions_list: List) -> List:
    """Given regions list. Will return the coordinates of all regions in the list."""
    coords_list = []
    for region in regions_list:
        coords = extract_region_props(region)[5]
        coords_list.append(coords)
    return coords_list


def coords_to_mask(coords_list: List, array: np.ndarray) -> np.ndarray:
    """Given coordinates and template array. Will turn coordinates into a binary mask."""
    mask = np.zeros(array.shape)
    for coords in coords_list:
        for kk in range(0, coords.shape[0]):
            mask[coords[kk, 0], coords[kk, 1]] = 1
    return mask


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Given a mask. Will return an inverted mask."""
    invert_mask = mask == 0
    return invert_mask


def coords_to_inverted_mask(coords_list: List, array: np.ndarray) -> np.ndarray:
    """Given coordinates and template array. Will turn coordinates into an inverted binary mask."""
    mask = coords_to_mask(coords_list, array)
    inverted_mask = invert_mask(mask)
    return inverted_mask


def mask_to_contour(mask: np.ndarray) -> np.ndarray:
    """Given a mask of the wound. Will return contour around wound."""
    filter_size = 1
    blur = apply_gaussian_filter(mask, filter_size)
    # contours = measure.find_contours(blur, 0.95)
    contours = measure.find_contours(blur, 0.75)
    contour_list = []
    contour_leng = []
    for cont in contours:
        contour_list.append(cont)
        contour_leng.append(cont.shape[0])
    if len(contour_list) == 0:
        return None
    else:
        argmax = np.argmax(contour_leng)
        chosen_contour = contour_list[argmax]
        return chosen_contour


def close_region(array: np.ndarray, radius: int = 1) -> np.ndarray:
    """Given an array with a small hole. Will return a closed array."""
    footprint = morphology.disk(radius, dtype=bool)
    closed_array = morphology.binary_closing(array, footprint)
    return closed_array


def dilate_region(array: np.ndarray, radius: int = 1) -> np.ndarray:
    """Given an array with a small hole. Will return a closed array."""
    footprint = morphology.disk(radius, dtype=bool)
    dilated_array = morphology.binary_dilation(array, footprint)
    return dilated_array


def gabor_filter(array: np.ndarray, theta_range: int = 17, ff_max: int = 11, ff_mult: float = 0.1) -> np.ndarray:
    gabor_all = np.zeros(array.shape)
    for ff in range(0, ff_max):
        frequency = 0.2 + ff * ff_mult
        for tt in range(0, theta_range):
            theta = tt * np.pi / (theta_range - 1)
            filt_real, _ = gabor(array, frequency=frequency, theta=theta)
            gabor_all += filt_real
    return gabor_all


def threshold_array(array: np.ndarray, selection_idx: int) -> np.ndarray:
    """Given an image wrray. Will return a binary array where object = 0, background = 1."""
    if selection_idx == 1:
        """Given a brightfield image array. Will return a binary array where tissue = 0, background = 1."""
        median_filter_size = 5
        array_median = apply_median_filter(array, median_filter_size)
        gaussian_filter_size = 2
        array_gaussian = apply_gaussian_filter(array_median, gaussian_filter_size)
        thresh_img = apply_otsu_thresh(array_gaussian)
        return thresh_img
    elif selection_idx == 2:
        """Given a gfp image array. Will return a binary array where gfp = 0, background = 1."""
        median_filter_size = 5
        array_median = apply_median_filter(array, median_filter_size)
        gaussian_filter_size = 1
        array_gaussian = apply_gaussian_filter(array_median, gaussian_filter_size)
        thresh_img = apply_otsu_thresh(array_gaussian)
        thresh_img_inverted = invert_mask(thresh_img)
        return thresh_img_inverted
    elif selection_idx == 3:
        """Given a phase contrast ph1 image array. Will return a binary array where tissue = 0, background = 1."""
        gabor_all = gabor_filter(array)
        thresh_img = apply_otsu_thresh(gabor_all)
        thresh_img_inverted = invert_mask(thresh_img)
        return thresh_img_inverted
    elif selection_idx == 4:
        """Given a phase contrast ph1 image array. Will return a binary array where tissue = 0, background = 1."""
        gabor_all = gabor_filter(array)
        median_filter_size = 5
        median_applied = apply_median_filter(gabor_all, median_filter_size)
        gaussian_filter_size = 2
        gaussian_applied = apply_gaussian_filter(median_applied, gaussian_filter_size)
        thresh_img = apply_otsu_thresh(gaussian_applied)
        thresh_img_inverted = invert_mask(thresh_img)
        return thresh_img_inverted
    else:
        raise ValueError("specified version is not supported")


def preview_thresholding(img: np.ndarray) -> list:
    """Given an image array. Will run all thresholds on the array for preview."""
    thresh_list = []
    idx_list = []
    for kk in range(1, 5):
        thresh_list.append(threshold_array(img, kk))
        idx_list.append(kk)
    return thresh_list, idx_list


def get_mean_center(array: np.ndarray) -> Union[float, int]:
    """ """
    coords = np.argwhere(array > 0)
    center_0 = np.mean(coords[:, 0])
    center_1 = np.mean(coords[:, 1])
    return center_0, center_1


def isolate_masks(array: np.ndarray, selection_idx: int) -> np.ndarray:
    """Given a binary mask where background = 1. Will return a mask where `tissue' = 1.
    Will return a mask where `wound' = 1."""
    if selection_idx == 1 or selection_idx == 2 or selection_idx == 3 or selection_idx == 4:
        # select the three largest "background" regions -- side, side, wound
        region_props = get_region_props(array)
        # new approach -> remove all regions that aren't touching the boundaries
        region_props_not_touching = get_regions_not_touching_bounds(region_props, array.shape)
        num_regions = 10  # changed from 3
        regions_largest = get_largest_regions(region_props, num_regions)
        num_regions = 1
        region_not_touching_largest = get_largest_regions(region_props_not_touching, num_regions)
        # identify the wound as the "background" region closest to the center
        array_inverted = invert_mask(array)
        center_0, center_1 = get_mean_center(array_inverted)
        # center_0, center_1 = get_domain_center(array)
        if len(region_not_touching_largest) > 0:
            # create the wound mask
            wound_region = get_closest_region(region_not_touching_largest, center_0, center_1)
            wound_region_coords = region_to_coords([wound_region])
            wound_mask_open = coords_to_mask(wound_region_coords, array)
            wound_mask = close_region(wound_mask_open)
        else:
            wound_mask = np.zeros(array.shape)
            wound_region = None
        # create the tissue mask
        regions_largest_coords = region_to_coords(regions_largest)
        tissue_mask_extra = coords_to_inverted_mask(regions_largest_coords, array)
        region_props = get_region_props(tissue_mask_extra)
        num_regions = 1
        regions_largest = get_largest_regions(region_props, num_regions)
        regions_largest_coords = region_to_coords(regions_largest)
        tissue_mask_open = coords_to_mask(regions_largest_coords, array)
        tissue_mask = close_region(tissue_mask_open)
        return tissue_mask, wound_mask, wound_region
    else:
        raise ValueError("specified version is not supported")


def threshold_all(img_list: List, threshold_function_idx: int) -> List:
    """Given an image list and function index. Will apply threshold to all images."""
    thresholded_list = []
    for img in img_list:
        thresh_img = threshold_array(img, threshold_function_idx)
        thresholded_list.append(thresh_img)
    return thresholded_list


def mask_all(thresh_img_list: List, selection_idx: int) -> List:
    """Given a thresholded image list. Will return masks and wound regions."""
    tissue_mask_list = []
    wound_mask_list = []
    wound_region_list = []
    for thresh_img in thresh_img_list:
        if selection_idx == 4:
            tissue_mask, wound_mask, wound_region = isolate_masks(thresh_img, selection_idx)
        else:
            _, wound_mask, wound_region = isolate_masks(thresh_img, selection_idx)
            tissue_mask, _, _ = isolate_masks(thresh_img, 4)
        tissue_mask_list.append(tissue_mask)
        wound_mask_list.append(wound_mask)
        wound_region_list.append(wound_region)
    return tissue_mask_list, wound_mask_list, wound_region_list


def check_above_min_size(region: object, min_area: Union[int, float]):
    """Will check if region is above a minimum area."""
    wound_area, _, _, _, _, _, _, _ = extract_region_props(region)
    if wound_area > min_area:
        return True
    else:
        return False


def contour_all(wound_mask_list: List) -> List:
    """Given a wound mask list. Will return a contour list."""
    contour_list = []
    for wound_mask in wound_mask_list:
        contour = mask_to_contour(wound_mask)
        contour_list.append(contour)
    return contour_list


def fill_tissue_mask_reconstruction(mask: np.ndarray) -> np.ndarray:
    """Given a tissue mask. Will return a filled tissue mask w/ reconstruction."""
    gaussian_filter_size = 1
    mask_gaussian = apply_gaussian_filter(mask, gaussian_filter_size)
    new_mask = mask_gaussian > 0
    seed = np.copy(new_mask)
    seed[1:-1, 1:-1] = new_mask.max()
    reconstruction_mask = new_mask
    mask_filled = morphology.reconstruction(seed, reconstruction_mask, method='erosion')
    return mask_filled


def insert_borders(mask: np.ndarray, border: int = 10) -> np.ndarray:
    """Given a mask. Will make the borders around it 0."""
    mask[0:border, :] = 0
    mask[-border:, :] = 0
    mask[:, 0:border] = 0
    mask[:, -border:] = 0
    return mask


def make_tissue_mask_robust(tissue_mask: np.ndarray, wound_mask: np.ndarray) -> np.ndarray:
    """Given a tissue mask and wound mask. Will fill holes in the tissue mask and make it suitable
    for computing the tissue contour etc."""
    tissue_mask_filled_1 = tissue_mask + wound_mask
    tissue_mask_filled_2 = fill_tissue_mask_reconstruction(tissue_mask_filled_1)
    tissue_mask_filled_3 = apply_gaussian_filter(tissue_mask_filled_2, 1) > 0
    tissue_mask_borders = insert_borders(tissue_mask_filled_3)
    # return largest connected component -- TODO: make sure this doesn't mess up other tests
    regions = get_region_props(tissue_mask_borders)
    largest_regions = get_largest_regions(regions, 1)
    coords = region_to_coords(largest_regions)
    tissue_mask_robust = coords_to_mask(coords, tissue_mask_borders)
    return tissue_mask_robust


def select_threshold_function(
    input_dict: dict,
    is_brightfield: bool,
    is_fluorescent: bool,
    is_ph1: bool
) -> int:
    """Given setup information. Will return which segmentation function to run."""
    if is_brightfield and input_dict["seg_bf_version"] == 1:
        return 1
    elif is_fluorescent and input_dict["seg_fl_version"] == 1:
        return 2
    elif is_ph1 and input_dict["seg_ph1_version"] == 1:
        return 3
    elif is_ph1 and input_dict["seg_ph1_version"] == 2:
        return 4
    else:
        raise ValueError("specified version is not supported")