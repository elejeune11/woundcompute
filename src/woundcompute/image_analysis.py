import alphashape
import cv2
import glob
import imageio.v2 as imageio
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy import ndimage
from scipy.interpolate import splprep, splev
from scipy.spatial import distance
from shapely.geometry import Point
from skimage import exposure, img_as_ubyte
from skimage import io
from skimage import measure, morphology
from skimage.filters import threshold_otsu, gabor
from skimage.measure import label, regionprops
import time
from typing import List, Union
import yaml


def hello_wound_compute() -> str:
    "Given no input. Simple hello world as a test function."
    return "Hello World!"


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


def mask_to_area(mask: np.ndarray, pix_to_microns: Union[float, int] = 1):
    """Given a mask and pixel to micron conversions. Returns wound area."""
    area = np.sum(mask)
    area_scaled = area * pix_to_microns * pix_to_microns
    return area_scaled


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


def read_tiff(img_path: Path) -> np.ndarray:
    """Given a path to a tiff. Will return an array."""
    img = io.imread(img_path)
    return img


def uint16_to_uint8(img_16: np.ndarray) -> np.ndarray:
    """Given a uint16 image. Will normalize + rescale and convert to uint8."""
    img_8 = img_as_ubyte(exposure.rescale_intensity(img_16))
    return img_8


def show_and_save_image(img_array: np.ndarray, save_path: Path, title: str = 'no_title') -> None:
    """Given an image and path location. Will plot and save image."""
    if title == 'no_title':
        plt.imsave(save_path, img_array, cmap=plt.cm.gray)
    else:
        plt.figure()
        plt.imshow(img_array, cmap=plt.cm.gray)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    return


def show_and_save_contour(
    img_array: np.ndarray,
    contour: np.ndarray,
    is_broken: bool,
    is_closed: bool,
    save_path: Path,
    title: str = " "
) -> None:
    """Given an image, contour, and path location. Will plot and save."""
    plt.figure()
    plt.imshow(img_array, cmap=plt.cm.gray)
    xt = 3.0 * img_array.shape[1] / 8.0
    yt = 7.0 * img_array.shape[0] / 8.0
    if is_broken:
        plt.text(xt, yt, "broken", color="r", backgroundcolor="w", fontsize=20)
    else:
        if is_closed:
            plt.text(xt, yt, "closed", color="r", backgroundcolor="w", fontsize=20)
        else:
            if contour is not None:
                plt.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2.0, antialiased=True)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return


def show_and_save_contour_and_width(
    img_array: np.ndarray,
    contour: np.ndarray,
    is_broken: bool,
    is_closed: bool,
    points: List,
    save_path: Path,
    title: str = " "
) -> None:
    """Given an image, contour, and path location. Will plot and save."""
    plt.figure()
    plt.imshow(img_array, cmap=plt.cm.gray)
    xt = 3.0 * img_array.shape[1] / 8.0
    yt = 7.0 * img_array.shape[0] / 8.0
    if is_broken:
        plt.text(xt, yt, "broken", color="r", backgroundcolor="w", fontsize=20)
    else:
        if points is not None:
            plt.plot(points[1], points[0], 'k-o', linewidth=2.0, antialiased=True)
        if is_closed:
            plt.text(xt, yt, "closed", color="r", backgroundcolor="w", fontsize=20)
        else:
            if contour is not None:
                plt.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2.0, antialiased=True)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return


def show_and_save_double_contour(
    img_array: np.ndarray,
    contour_bf: np.ndarray,
    contour_fl: np.ndarray,
    save_path: Path,
    title: str = " "
) -> None:
    """Given an image, contour, and path location. Will plot and save."""
    plt.figure()
    plt.imshow(img_array, cmap=plt.cm.gray)
    if contour_bf is not None:
        plt.plot(contour_bf[:, 1], contour_bf[:, 0], 'r', linewidth=2.0, antialiased=True)
    if contour_fl is not None:
        plt.plot(contour_fl[:, 1], contour_fl[:, 0], 'c:', linewidth=2.0, antialiased=True)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return


def save_numpy(array: np.ndarray, save_path: Path) -> None:
    """Given a numpy array and path location. Will save as numpy array."""
    if np.all(array == (array > 0)):
        np.save(save_path, array > 0)
    else:
        np.save(save_path, array)
    return


def _yml_to_dict(*, yml_path_file: Path) -> dict:
    """Given a valid Path to a yml input file, read it in and
    return the result as a dictionary."""

    # Compared to the lower() method, the casefold() method is stronger.
    # It will convert more characters into lower case, and will find more matches
    # on comparison of two strings that are both are converted
    # using the casefold() method.
    woundcompute: str = "woundcompute>"

    if not yml_path_file.is_file():
        raise FileNotFoundError(f"{woundcompute} File not found: {str(yml_path_file)}")

    file_type = yml_path_file.suffix.casefold()

    supported_types = (".yaml", ".yml")

    if file_type not in supported_types:
        raise TypeError("Only file types .yaml, and .yml are supported.")

    try:
        with open(yml_path_file, "r") as stream:
            # See deprecation warning for plain yaml.load(input) at
            # https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
            db = yaml.load(stream, Loader=yaml.SafeLoader)
    except yaml.YAMLError as error:
        print(f"Error with YAML file: {error}")
        # print(f"Could not open: {self.self.path_file_in}")
        print(f"Could not open or decode: {yml_path_file}")
        # raise yaml.YAMLError
        raise OSError

    version_specified = db.get("version")
    version_implemented = 1.0

    if version_specified != version_implemented:
        raise ValueError(
            f"Version mismatch: specified was {version_specified}, implemented is {version_implemented}"
        )
    else:
        # require that input file has at least the following keys:
        required_keys = (
            "version",
            "segment_brightfield",
            "seg_bf_version",
            "seg_bf_visualize",
            "segment_fluorescent",
            "seg_fl_version",
            "seg_fl_visualize",
            "segment_ph1",
            "seg_ph1_version",
            "seg_ph1_visualize",
            "track_brightfield",
            "track_bf_version",
            "track_bf_visualize",
            "track_ph1",
            "track_ph1_version",
            "track_ph1_visualize",
            "bf_seg_with_fl_seg_visualize",
            "bf_track_with_fl_seg_visualize",
            "ph1_seg_with_fl_seg_visualize",
            "ph1_track_with_fl_seg_visualize",
        )

        # has_required_keys = all(tuple(map(lambda x: db.get(x) != None, required_keys)))
        # keys_tuple = tuple(map(lambda x: db.get(x), required_keys))
        # has_required_keys = all(tuple(map(lambda x: db.get(x), required_keys)))
        found_keys = tuple(db.keys())
        keys_exist = tuple(map(lambda x: x in found_keys, required_keys))
        has_required_keys = all(keys_exist)
        if not has_required_keys:
            raise KeyError(f"Input files must have these keys defined: {required_keys}")
    return db


def create_folder(folder_path: Path, new_folder_name: str) -> Path:
    """Given a path to a directory and a folder name. Will create a directory in the given directory."""
    new_path = folder_path.joinpath(new_folder_name).resolve()
    if new_path.exists() is False:
        os.mkdir(new_path)
    return new_path


def input_info_to_input_dict(folder_path: Path) -> dict:
    """Given a folder path that contains a yaml file. Will return the input dictionary."""
    yaml_name_list = glob.glob(str(folder_path) + '/*.yaml') + glob.glob(str(folder_path) + '/*.yml')
    yml_path_file = Path(yaml_name_list[0])
    input_dict = _yml_to_dict(yml_path_file=yml_path_file)
    return input_dict


def input_info_to_input_paths(folder_path: Path) -> dict:
    """Given a folder path. Will return the path to the image folders."""
    path_dict = {}
    bf_path = folder_path.joinpath("brightfield_images").resolve()
    if bf_path.is_dir():
        path_dict["brightfield_images_path"] = bf_path
    else:
        path_dict["brightfield_images_path"] = None
    fl_path = folder_path.joinpath("fluorescent_images").resolve()
    if fl_path.is_dir():
        path_dict["fluorescent_images_path"] = fl_path
    else:
        path_dict["fluorescent_images_path"] = None
    ph1_path = folder_path.joinpath("ph1_images").resolve()
    if ph1_path.is_dir():
        path_dict["ph1_images_path"] = ph1_path
    else:
        path_dict["ph1_images_path"] = None
    return path_dict


def image_folder_to_path_list(folder_path: Path) -> List:
    """Given a folder path. Will return the path to all files in that path in order."""
    name_list = glob.glob(str(folder_path) + '/*.TIF')
    name_list.sort()
    name_list_path = []
    for name in name_list:
        name_list_path.append(Path(name))
    return name_list_path


def input_info_to_output_paths(folder_path: Path, input_dict: dict) -> dict:
    """Given a path to a directory and the input information. Will create output directories."""
    path_dict = {}
    if input_dict["segment_brightfield"] is True:
        segment_brightfield_path = create_folder(folder_path, "segment_brightfield")
        path_dict["segment_brightfield_path"] = segment_brightfield_path
    else:
        path_dict["segment_brightfield_path"] = None
    if input_dict["seg_bf_visualize"] is True:
        segment_brightfield_vis_path = create_folder(segment_brightfield_path, "visualizations")
        path_dict["segment_brightfield_vis_path"] = segment_brightfield_vis_path
    else:
        path_dict["segment_brightfield_vis_path"] = None
    if input_dict["segment_fluorescent"] is True:
        segment_fluorescent_path = create_folder(folder_path, "segment_fluorescent")
        path_dict["segment_fluorescent_path"] = segment_fluorescent_path
    else:
        path_dict["segment_fluorescent_path"] = None
    if input_dict["seg_fl_visualize"] is True:
        segment_fluorescent_vis_path = create_folder(segment_fluorescent_path, "visualizations")
        path_dict["segment_fluorescent_vis_path"] = segment_fluorescent_vis_path
    else:
        path_dict["segment_fluorescent_vis_path"] = None
    if input_dict["segment_ph1"] is True:
        segment_ph1_path = create_folder(folder_path, "segment_ph1")
        path_dict["segment_ph1_path"] = segment_ph1_path
    else:
        path_dict["segment_ph1_path"] = None
    if input_dict["seg_ph1_visualize"] is True:
        segment_ph1_vis_path = create_folder(segment_ph1_path, "visualizations")
        path_dict["segment_ph1_vis_path"] = segment_ph1_vis_path
    else:
        path_dict["segment_ph1_vis_path"] = None
    if input_dict["track_brightfield"] is True:
        track_brightfield_path = create_folder(folder_path, "track_brightfield")
        path_dict["track_brightfield_path"] = track_brightfield_path
    else:
        path_dict["track_brightfield_path"] = None
    if input_dict["track_bf_visualize"] is True:
        track_brightfield_vis_path = create_folder(track_brightfield_path, "visualizations")
        path_dict["track_brightfield_vis_path"] = track_brightfield_vis_path
    else:
        path_dict["track_brightfield_vis_path"] = None
    if input_dict["track_ph1"] is True:
        track_ph1_path = create_folder(folder_path, "track_ph1")
        path_dict["track_ph1_path"] = track_ph1_path
    else:
        path_dict["track_ph1_path"] = None
    if input_dict["track_ph1_visualize"] is True:
        track_ph1_vis_path = create_folder(track_ph1_path, "visualizations")
        path_dict["track_ph1_vis_path"] = track_ph1_vis_path
    else:
        path_dict["track_ph1_vis_path"] = None
    if input_dict["bf_seg_with_fl_seg_visualize"] is True:
        bf_seg_with_fl_seg_visualize_path = create_folder(folder_path, "bf_seg_with_fl_seg_visualize")
        path_dict["bf_seg_with_fl_seg_visualize_path"] = bf_seg_with_fl_seg_visualize_path
    else:
        path_dict["bf_seg_with_fl_seg_visualize_path"] = None
    if input_dict["bf_track_with_fl_seg_visualize"] is True:
        bf_track_with_fl_seg_visualize_path = create_folder(folder_path, "bf_track_with_fl_seg_visualize")
        path_dict["bf_track_with_fl_seg_visualize_path"] = bf_track_with_fl_seg_visualize_path
    else:
        path_dict["bf_track_with_fl_seg_visualize_path"] = None
    if input_dict["ph1_seg_with_fl_seg_visualize"] is True:
        ph1_seg_with_fl_seg_visualize_path = create_folder(folder_path, "ph1_seg_with_fl_seg_visualize")
        path_dict["ph1_seg_with_fl_seg_visualize_path"] = ph1_seg_with_fl_seg_visualize_path
    else:
        path_dict["ph1_seg_with_fl_seg_visualize_path"] = None
    if input_dict["ph1_track_with_fl_seg_visualize"] is True:
        ph1_track_with_fl_seg_visualize_path = create_folder(folder_path, "ph1_track_with_fl_seg_visualize")
        path_dict["ph1_track_with_fl_seg_visualize_path"] = ph1_track_with_fl_seg_visualize_path
    else:
        path_dict["ph1_track_with_fl_seg_visualize_path"] = None
    return path_dict


def input_info_to_dicts(folder_path: Path) -> dict:
    """Given a folder path. Will get input and output dictionaries set up."""
    input_dict = input_info_to_input_dict(folder_path)
    input_path_dict = input_info_to_input_paths(folder_path)
    output_path_dict = input_info_to_output_paths(folder_path, input_dict)
    return input_dict, input_path_dict, output_path_dict


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


def read_all_tiff(folder_path: Path) -> List:
    """Given a folder path. Will return a list of all tiffs as an array."""
    path_list = image_folder_to_path_list(folder_path)
    tiff_list = []
    for path in path_list:
        array = read_tiff(path)
        tiff_list.append(array)
    return tiff_list


def uint16_to_uint8_all(img_list: List) -> List:
    """Given an image list of uint16. Will return the same list all as uint8."""
    uint8_list = []
    for img in img_list:
        img8 = uint16_to_uint8(img)
        uint8_list.append(img8)
    return uint8_list


def save_all_numpy(folder_path: Path, file_name: str, array_list: List) -> None:
    """Given a folder path, file name, and array list. Will save the array as individual numpy arrays"""
    file_name_list = []
    for kk in range(0, len(array_list)):
        save_path = folder_path.joinpath(file_name + "_%05d.npy" % (kk)).resolve()
        file_name_list.append(save_path)
        if array_list[kk] is None:
            continue  # will not save an empty array
        else:
            save_numpy(array_list[kk], save_path)
    return file_name_list


def save_all_img_with_contour(
    folder_path: Path,
    file_name: str,
    img_list: List,
    contour_list: List,
    is_broken_list: List,
    is_closed_list: List
) -> List:
    "Given segmentation results. Plot and save image and contour."
    file_name_list = []
    for kk in range(0, len(img_list)):
        img = img_list[kk]
        cont = contour_list[kk]
        is_broken = is_broken_list[kk]
        is_closed = is_closed_list[kk]
        save_path = folder_path.joinpath(file_name + "_%05d.png" % (kk)).resolve()
        title = "frame %05d" % (kk)
        show_and_save_contour(img, cont, is_broken, is_closed, save_path, title)
        file_name_list.append(save_path)
    return file_name_list


def save_all_img_with_contour_and_width(
    folder_path: Path,
    file_name: str,
    img_list: List,
    contour_list: List,
    tissue_parameters_list: List,
    is_broken_list: List,
    is_closed_list: List
) -> List:
    "Given segmentation results. Plot and save image and contour."
    file_name_list = []
    for kk in range(0, len(img_list)):
        img = img_list[kk]
        cont = contour_list[kk]
        is_broken = is_broken_list[kk]
        is_closed = is_closed_list[kk]
        #  area, pt1_0, pt1_1, pt2_0, pt2_1, width, kappa_1, kappa_2
        tp = tissue_parameters_list[kk]
        points = [[tp[1], tp[3]], [tp[2], tp[4]]]
        save_path = folder_path.joinpath(file_name + "_%05d.png" % (kk)).resolve()
        title = "frame %05d" % (kk)
        show_and_save_contour_and_width(img, cont, is_broken, is_closed, points, save_path, title)
        file_name_list.append(save_path)
    return file_name_list


def save_all_img_with_double_contour(
    folder_path: Path,
    file_name: str,
    img_list: List,
    contour_list_bf: List,
    contour_list_fl: List
) -> List:
    "Given segmentation results. Plot and save image and contour."
    file_name_list = []
    for kk in range(0, len(img_list)):
        img = img_list[kk]
        cont_bf = contour_list_bf[kk]
        cont_fl = contour_list_fl[kk]
        save_path = folder_path.joinpath(file_name + "_%05d.png" % (kk)).resolve()
        title = "frame %05d" % (kk)
        show_and_save_double_contour(img, cont_bf, cont_fl, save_path, title)
        file_name_list.append(save_path)
    return file_name_list


def create_gif(folder_path: Path, file_name: str, file_list: List) -> Path:
    """Given a list of files. Creates a gif."""
    image_list = []
    for file in file_list:
        image_list.append(imageio.imread(file))
    gif_name = folder_path.joinpath(file_name + '.gif')
    imageio.mimsave(str(gif_name), image_list)
    return gif_name


def save_list(folder_path: Path, file_name: str, value_list: List):
    """Given a folder path, file name, and array list. Will save the array as a numpy array"""
    for kk in range(0, len(value_list)):
        if value_list[kk] is None:
            value_list[kk] = 0
    array = np.asarray(value_list)
    file_path = folder_path.joinpath(file_name + '.txt').resolve()
    np.savetxt(file_path, array)
    return file_path


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


def ix_loop(val: int, num_pts_contour: int) -> int:
    """Given an index value. Will loop it around (for contours)."""
    if val < 0:
        val = num_pts_contour + val
    if val >= num_pts_contour:
        val = val - num_pts_contour
    else:
        val = val
    return val


def get_contour_distance_across(
    c_idx: int,
    contour: np.ndarray,
    num_pts_contour: int,
    include_idx: List,
    tolerence_check: Union[float, int] = 0.2
) -> Union[float, int]:
    """Given a contour point and associated information. Will return the distance across the contour."""
    opposite_point = c_idx + int(num_pts_contour / 2)
    min_opposite = opposite_point - int(tolerence_check * num_pts_contour)
    max_opposite = opposite_point + int(tolerence_check * num_pts_contour)
    x0 = []
    x1 = []
    val_list = []
    for val_ix in range(min_opposite, max_opposite):
        val = ix_loop(val_ix, num_pts_contour)
        x0.append(contour[val, 0])
        x1.append(contour[val, 1])
        val_list.append(val)
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)
    x0_pt = contour[c_idx, 0]
    x1_pt = contour[c_idx, 1]
    dist_list = []
    for kk in range(0, x0.shape[0]):
        # if math.isinf(x0[kk]) or math.isinf(x1[kk]) or math.isinf(x0_pt) or math.isinf(x1_pt):
        #     dist_list.append(math.inf)
        # else:
        #     dist_list.append((x0[kk] - x0_pt) ** 2.0 + (x1[kk] - x1_pt) ** 2.0) ** 0.5
        if val_list[kk] in include_idx and c_idx in include_idx:
            dist = compute_distance(x0[kk], x1[kk], x0_pt, x1_pt)
            dist_list.append(dist)
        else:
            dist_list.append(math.inf)
    dist_array = np.asarray(dist_list)
    ix = np.argmin(dist_array)
    distance_opposite = dist_array[ix]
    ix_opposite = val_list[ix]
    return distance_opposite, ix_opposite


# def get_contour_distance_across_all(contour: np.ndarray) -> np.ndarray:
#     """Given a contour. Will compute the distance across the contour at every point."""
#     num_pts_contour = contour.shape[0]
#     tolerence_check = 0.2
#     distance_all = []
#     ix_all = []
#     for kk in range(0, num_pts_contour):
#         dist, ix = get_contour_distance_across(kk, contour, num_pts_contour, tolerence_check)
#         if math.isnan(dist):
#             distance_all.append(math.inf)
#         else:
#             distance_all.append(dist)
#         ix_all.append(ix)
#     distance_all = np.asarray(distance_all)
#     ix_all = np.asarray(ix_all)
#     return distance_all, ix_all

def get_contour_distance_across_all(contour: np.ndarray, include_idx: List) -> np.ndarray:
    """Given a contour and an include index. Will compute the distance across."""
    num_pts_contour = contour.shape[0]
    tolerence_check = 0.2
    distance_all = []
    ix_all = []
    for kk in range(0, num_pts_contour):
        if kk in include_idx:
            dist, ix = get_contour_distance_across(kk, contour, num_pts_contour, include_idx, tolerence_check)
        else:
            dist = math.inf
            ix = 0
        distance_all.append(dist)
        ix_all.append(ix)
    distance_all = np.asarray(distance_all)
    ix_all = np.asarray(ix_all)
    return distance_all, ix_all


def line_param(
    centroid_row: Union[float, int],
    centroid_col: Union[float, int],
    orientation: Union[float, int]
) -> Union[float, int]:
    """Given a point and a slope (orientation). Will return line format as ax_0 + bx_1 + c = 0."""
    line_a = -1.0 * np.tan(orientation)
    line_b = 1.0
    line_c = -1.0 * centroid_row + np.tan(orientation) * centroid_col
    return line_a, line_b, line_c


def dist_to_line(
    line_a: Union[float, int],
    line_b: Union[float, int],
    line_c: Union[float, int],
    pt_0: Union[float, int],
    pt_1: Union[float, int]
) -> Union[float, int]:
    """Given line parameters and a point. Will return the distance to the line."""
    numer = np.abs(line_a * pt_0 + line_b * pt_1 + line_c)
    denom = ((line_a) ** 2.0 + (line_b) ** 2.0) ** 0.5
    line_dist = numer / denom
    return line_dist


def move_point(
    pt_0: Union[float, int],
    pt_1: Union[float, int],
    line_a: Union[float, int],
    line_b: Union[float, int],
    line_c: Union[float, int],
    cutoff: Union[float, int]
) -> Union[float, int]:
    """Given a point and a line. Will move the point to the cutoff."""
    line_dist = dist_to_line(line_a, line_b, line_c, pt_0, pt_1)
    if np.abs(line_dist) < 10 ** -6:
        return pt_0, pt_1
    if line_a == 0:
        sig = pt_1 - (line_c / line_b)
        unit_vec_0 = -1.0 * np.sign(sig)
        unit_vec_1 = 0
    elif line_b == 0:
        unit_vec_0 = 0
        sig = pt_0 - (line_c / line_a)
        unit_vec_1 = -1.0 * np.sign(sig)
    else:
        # line_0_numer = -1.0 * line_b / line_a * pt_0 + pt_1 + line_c / line_b
        # line_0_denom = -1.0 * (line_b / line_a + line_a / line_b)
        line_0_numer = line_c / line_b + pt_1 - line_b / line_a * pt_0
        line_0_denom = -1.0 * (line_a / line_b + line_b / line_a)
        line_0 = line_0_numer / line_0_denom
        line_1 = -1.0 * line_a / line_b * line_0 - line_c / line_b
        vec_0 = line_0 - pt_0
        vec_1 = line_1 - pt_1
        unit_vec_0 = vec_0 / line_dist
        unit_vec_1 = vec_1 / line_dist
    pt_0_mod = pt_0 + unit_vec_0 * (line_dist - cutoff)
    pt_1_mod = pt_1 + unit_vec_1 * (line_dist - cutoff)
    return pt_0_mod, pt_1_mod


def clip_contour(
    contour: np.ndarray,
    centroid_row: Union[int, float],
    centroid_col: Union[int, float],
    orientation: Union[int, float],
    tissue_axis_major_length: Union[int, float],
    tissue_axis_minor_length: Union[int, float]
) -> np.ndarray:
    cutoff = tissue_axis_major_length / 3.0
    line_a, line_b, line_c = line_param(centroid_row, centroid_col, orientation)
    contour_clipped = []
    for kk in range(0, contour.shape[0]):
        pt_0 = contour[kk, 0]
        pt_1 = contour[kk, 1]
        line_dist = dist_to_line(line_a, line_b, line_c, pt_1, pt_0)
        if line_dist < cutoff:
            contour_clipped.append([pt_0, pt_1])
        else:
            pt_1_mod, pt_0_mod = move_point(pt_1, pt_0, line_a, line_b, line_c, cutoff)
            contour_clipped.append([pt_0_mod, pt_1_mod])
    # if len(contour_clipped) < contour.shape[0]:
    #     contour_clipped.append(contour_clipped[0])
    contour_clipped = np.asarray(contour_clipped)
    return contour_clipped


def include_points_contour(
    contour: np.ndarray,
    centroid_row: Union[int, float],
    centroid_col: Union[int, float],
    tissue_axis_major_length: Union[int, float],
    tissue_axis_minor_length: Union[int, float]
) -> List:
    """Given information about the tissue contour. Will return included points for tissue width."""
    # radius = 0.25 * (tissue_axis_major_length + tissue_axis_minor_length)
    radius = 0.5 * tissue_axis_minor_length
    include_idx = []
    for kk in range(0, contour.shape[0]):
        dist = compute_distance(contour[kk, 0], contour[kk, 1], centroid_row, centroid_col)
        if dist < radius:
            include_idx.append(kk)
    return include_idx


def resample_contour(contour: np.ndarray) -> np.ndarray:
    """Given a contour. Will resample and return the resampled contour."""
    ix_0 = contour[:, 0]
    ix_1 = contour[:, 1]
    tck, u = splprep([ix_0, ix_1], s=0)
    resampled_contour_list = splev(u, tck)
    resampled_contour = np.asarray(resampled_contour_list).T
    num_pts_max = 250
    num = np.max([int(resampled_contour.shape[0] / num_pts_max), 1])
    downsampled_contour = resampled_contour[::num, :]
    return downsampled_contour


def get_penalized(contour: np.ndarray, contour_clipped: np.ndarray):
    """Given the original contour and the clipped contour. Will return penalized contour."""
    cc_penalized = []
    for kk in range(0, contour.shape[0]):
        if math.isclose(contour[kk, 0], contour_clipped[kk, 0]) and math.isclose(contour[kk, 1], contour_clipped[kk, 1]):
            cc_penalized.append([contour[kk, 0], contour[kk, 1]])
        else:
            cc_penalized.append([math.inf, math.inf])
    cc_penalized = np.asarray(cc_penalized)
    return cc_penalized


def get_contour_width(
    contour: np.ndarray,
    centroid_row: Union[int, float],
    centroid_col: Union[int, float],
    tissue_axis_major_length: Union[int, float],
    tissue_axis_minor_length: Union[int, float],
    orientation: Union[int, float]
) -> Union[float, int]:
    """Given a contour. Will compute minimum distance across and location of minimum. This is the width."""
    include_idx = include_points_contour(contour, centroid_row, centroid_col, tissue_axis_major_length, tissue_axis_minor_length)
    # contour_clipped_0 = clip_contour(contour, centroid_row, centroid_col, orientation, tissue_axis_major_length, tissue_axis_minor_length)
    # contour_clipped = clip_contour(contour_clipped_0, centroid_row, centroid_col, orientation + np.pi / 2.0, tissue_axis_major_length, tissue_axis_minor_length)
    # contour_clipped_penalized = get_penalized(contour, contour_clipped)
    # distance_all, ix_all = get_contour_distance_across_all(contour_clipped_penalized)
    distance_all, ix_all = get_contour_distance_across_all(contour, include_idx)
    idx_a = np.argmin(distance_all)
    width = distance_all[idx_a]
    idx_b = ix_all[idx_a]
    return width, idx_a, idx_b


# def get_contouor_distance_across_all_v2(contour: np.ndarray) -> np.ndarray:
#     """Given a contour. Will compute the distance across the contour at every point."""
#     return distance_all, ix_all


# def get_contour_width_v2(contour: np.ndarray) -> Union[float, int]:
#     """Given a contour. Will compute minimum distance across and location of minimum. This is the width."""
#     return width, idx_a, idx_b


def get_local_curvature(contour: np.ndarray, mask: np.ndarray, ix_center: int, sample_dist: int) -> Union[float, int]:
    """Given a contour and a specified location. Will return curvature."""
    sample_dist = int(sample_dist)
    num_pts_contour = contour.shape[0]
    x0 = []
    x1 = []
    for kk in range(-sample_dist, sample_dist):
        val = ix_center + kk
        val = ix_loop(val, num_pts_contour)
        x0.append(contour[val, 0])
        x1.append(contour[val, 1])
    # find the best fit circle, see:
    # https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)
    # coordinate of the barycenter
    x0_m = np.mean(x0)
    x1_m = np.mean(x1)
    # find the sign of the curvature
    midpoint_0 = int(x0_m)
    midpoint_1 = int(x1_m)
    if mask[midpoint_0, midpoint_1] > 0:
        kappa_sign = 1.0
    else:
        kappa_sign = -1.0
    # calculate the reduced coordinates
    u = x0 - x0_m
    v = x1 - x1_m
    # linear system defining the center (uc, vc) in reduced coordinates:
    #   Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #   Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv = np.sum(u * v)
    Suu = np.sum(u ** 2)
    Svv = np.sum(v ** 2)
    Suuv = np.sum(u ** 2 * v)
    Suvv = np.sum(u * v ** 2)
    Suuu = np.sum(u ** 3)
    Svvv = np.sum(v ** 3)
    # Solving the linear system
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
    if np.abs(np.linalg.det(A)) < 10 ** np.finfo(float).eps * 10:
        kappa_correct_sign = float('inf')
    else:
        uc, vc = np.linalg.solve(A, B)
        x0c_1 = x0_m + uc
        x1c_1 = x1_m + vc
        # Calculates the distance from the center (xc_1, yc_1)
        Ri_1 = np.sqrt((x0 - x0c_1)**2 + (x1 - x1c_1)**2)
        R_1 = np.mean(Ri_1)
        # residu_1 = np.sum((Ri_1-R_1) ** 2)
        kappa = 1 / R_1
        kappa_correct_sign = kappa * kappa_sign
    return kappa_correct_sign


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
    tissue_mask_robust = insert_borders(tissue_mask_filled_3)
    return tissue_mask_robust


def tissue_parameters(tissue_mask: np.ndarray, wound_mask: np.ndarray) -> Union[float, int]:
    """Given a tissue mask. Will compute and return key properties."""
    area = np.sum(tissue_mask)
    tissue_mask_robust = make_tissue_mask_robust(tissue_mask, wound_mask)
    # tm_c_0, tm_c_1 = get_mean_center(tissue_mask_robust)
    tissue_contour = mask_to_contour(tissue_mask_robust)
    tissue_regions_all = get_region_props(tissue_mask_robust)
    tissue_region = get_largest_regions(tissue_regions_all, 1)[0]
    _, tissue_axis_major_length, tissue_axis_minor_length, centroid_row, centroid_col, _, _, orientation = extract_region_props(tissue_region)
    width, contour_idx_0, contour_idx_1 = get_contour_width(tissue_contour, centroid_row, centroid_col, tissue_axis_major_length, tissue_axis_minor_length, orientation)
    sample_dist = np.min([100, tissue_contour.shape[0] * 0.1])
    kappa_1 = get_local_curvature(tissue_contour, tissue_mask_robust, contour_idx_0, sample_dist)
    kappa_2 = get_local_curvature(tissue_contour, tissue_mask_robust, contour_idx_1, sample_dist)
    pt1_0 = tissue_contour[contour_idx_0, 0]
    pt1_1 = tissue_contour[contour_idx_0, 1]
    pt2_0 = tissue_contour[contour_idx_1, 0]
    pt2_1 = tissue_contour[contour_idx_1, 1]
    return width, area, kappa_1, kappa_2, pt1_0, pt1_1, pt2_0, pt2_1, tissue_contour


# def tissue_parameters_all(tissue_mask_list: np.ndarray, wound_mask_list: np.ndarray) -> List:
#     """Given a tissue mask list. Will return tissue parameters list."""
#     tissue_width_list = []  # width at measurement location
#     tissue_area_list = []  # tissue area, will not be meaningful if not standardized
#     tissue_curvature_list = []  # kappa_1, kappa_2 at measurement locations
#     tissue_measurement_locations_list = []  # row_1, col_1, row_2, col_2
#     for kk in range(0,len(tissue_mask_list)):
#         tissue_mask = tissue_mask_list[kk]
#         wound_mask = wound_mask_list[kk]
#         width, area, kappa_1, kappa_2, pt1_0, pt1_1, pt2_0, pt2_1, tissue_contour = tissue_parameters(tissue_mask, wound_mask)
#         tissue_width_list.append(width)
#         tissue_area_list.append(area)
#         tissue_curvature_list.append([kappa_1, kappa_2])
#         tissue_measurement_locations_list.append([pt1_0, pt1_1, pt2_0, pt2_1])
#     return tissue_width_list, tissue_area_list, tissue_curvature_list, tissue_measurement_locations_list


def wound_parameters_all(wound_region_list: List) -> List:
    """Given a wound regions list. Will return wound properties list."""
    area_list = []
    axis_major_length_list = []
    axis_minor_length_list = []
    for wound_region in wound_region_list:
        area, axis_major_length, axis_minor_length, _, _, _, _, _ = extract_region_props(wound_region)
        area_list.append(area)
        axis_major_length_list.append(axis_major_length)
        axis_minor_length_list.append(axis_minor_length)
    return area_list, axis_major_length_list, axis_minor_length_list


def tissue_parameters_all(tissue_mask_list: List, wound_mask_list: List) -> List:
    """Given tissue and wound masks. Will return tissue parameters."""
    #  parameter list has order:
    #  area, pt1_0, pt1_1, pt2_0, pt2_1, width, kappa_1, kappa_2
    parameter_list = []
    for kk in range(0, len(tissue_mask_list)):
        width, area, kappa_1, kappa_2, pt1_0, pt1_1, pt2_0, pt2_1, _ = tissue_parameters(tissue_mask_list[kk], wound_mask_list[kk])
        param = [area, pt1_0, pt1_1, pt2_0, pt2_1, width, kappa_1, kappa_2]
        parameter_list.append(param)
    return parameter_list


def run_segment(input_path: Path, output_path: Path, threshold_function_idx: int) -> List:
    """Given input and output information. Will run the complete segmentation process."""
    # read the inputs
    img_list = read_all_tiff(input_path)
    # apply threshold
    thresholded_list = threshold_all(img_list, threshold_function_idx)
    # masking
    tissue_mask_list, wound_mask_list, wound_region_list = mask_all(thresholded_list, threshold_function_idx)
    # contour
    contour_list = contour_all(wound_mask_list)
    # wound parameters
    area_list, axis_major_length_list, axis_minor_length_list = wound_parameters_all(wound_region_list)
    # tissue parameters
    tissue_parameters_list = tissue_parameters_all(tissue_mask_list, wound_mask_list)
    # save numpy arrays
    wound_name_list = save_all_numpy(output_path, "wound_mask", wound_mask_list)
    tissue_name_list = save_all_numpy(output_path, "tissue_mask", tissue_mask_list)
    contour_name_list = save_all_numpy(output_path, "contour_coords", contour_list)
    # save lists
    area_path = save_list(output_path, "wound_area_vs_frame", area_list)
    ax_maj_path = save_list(output_path, "wound_major_axis_length_vs_frame", axis_major_length_list)
    ax_min_path = save_list(output_path, "wound_minor_axis_length_vs_frame", axis_minor_length_list)
    tissue_path = save_list(output_path, "tissue_parameters_vs_frame", tissue_parameters_list)
    # check if the tissue is broken
    is_broken_list = check_broken_tissue_all(tissue_mask_list)
    is_broken_path = save_list(output_path, "is_broken_vs_frame", is_broken_list)
    # check if the wound is closed
    is_closed_list = check_wound_closed_all(tissue_mask_list, wound_region_list)
    is_closed_path = save_list(output_path, "is_closed_vs_frame", is_closed_list)
    return wound_name_list, tissue_name_list, contour_name_list, area_path, ax_maj_path, ax_min_path, tissue_path, is_broken_path, is_closed_path, img_list, contour_list, tissue_parameters_list, is_broken_list, is_closed_list


def get_tracking_param_dicts() -> dict:
    """Will return dictionaries specifying the feature parameters and tracking parameters.
    In future, these may vary based on version."""
    feature_params = dict(maxCorners=1000, qualityLevel=0.1, minDistance=7, blockSize=7)
    window = 50
    lk_params = dict(winSize=(window, window), maxLevel=10, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    return feature_params, lk_params


def bool_to_uint8(arr_bool: np.ndarray) -> np.ndarray:
    """Given a boolean array. Will return a uint8 array."""
    arr_uint8 = (1. * arr_bool).astype('uint8')
    return arr_uint8


def mask_to_track_points(img_uint8: np.ndarray, mask: np.ndarray, feature_params: dict) -> np.ndarray:
    """Given an image and a mask. Will return the good features to track within the mask region."""
    # ensure that the mask is uint8
    mask_uint8 = bool_to_uint8(mask)
    track_points_0 = cv2.goodFeaturesToTrack(img_uint8, mask=mask_uint8, **feature_params)
    return track_points_0


def track_one_step(img_uint8_0: np.ndarray, img_uint8_1: np.ndarray, track_points_0: np.ndarray, lk_params: dict):
    """Given img_0, img_1, tracking points p0, and tracking parameters.
    Will return the tracking points new location. Note that for now standard deviation and error are ignored."""
    track_points_1, _, _ = cv2.calcOpticalFlowPyrLK(img_uint8_0, img_uint8_1, track_points_0, None, **lk_params)
    return track_points_1


def get_order_track(len_img_list: int, is_forward: bool) -> List:
    """Given the length of the image list. Will return the order of tracking frames"""
    if is_forward:
        return list(range(0, len_img_list))
    else:
        return list(range(len_img_list - 1, -1, -1))


def track_all_steps(img_list_uint8: List, mask: np.ndarray, order_list: List) -> np.ndarray:
    """Given the image list, mask, and order. Will run tracking through the whole img list in order.
    Note that the returned order of tracked points will match order_list."""
    feature_params, lk_params = get_tracking_param_dicts()
    img_0 = img_list_uint8[order_list[0]]
    track_points = mask_to_track_points(img_0, mask, feature_params)
    num_track_pts = track_points.shape[0]
    num_imgs = len(img_list_uint8)
    tracker_x = np.zeros((num_track_pts, num_imgs))
    tracker_y = np.zeros((num_track_pts, num_imgs))
    for kk in range(0, num_imgs - 1):
        tracker_x[:, kk] = track_points[:, 0, 0]
        tracker_y[:, kk] = track_points[:, 0, 1]
        ix_0 = order_list[kk]
        ix_1 = order_list[kk + 1]
        img_0 = img_list_uint8[ix_0]
        img_1 = img_list_uint8[ix_1]
        track_points = track_one_step(img_0, img_1, track_points, lk_params)
    tracker_x[:, kk + 1] = track_points[:, 0, 0]
    tracker_y[:, kk + 1] = track_points[:, 0, 1]
    return tracker_x, tracker_y


def get_unique(numbers):
    """Helper function for getting unique values in a list."""
    list_of_unique_numbers = []
    unique_numbers = set(numbers)
    for number in unique_numbers:
        list_of_unique_numbers.append(number)
    return list_of_unique_numbers


def wound_mask_from_points(
    frame_0_mask: np.ndarray,
    tracker_x: np.ndarray,
    tracker_y: np.ndarray,
    wound_contour: np.ndarray,
    alpha_assigned: bool = True,
    assigned_alpha: float = 0.01
) -> np.ndarray:
    """Given tracking results and frame 0 wound contour. Will create wound masks based on the alphashape of the close track points."""
    num_pts = tracker_x.shape[0]
    final_xy = np.zeros((num_pts, 2))
    final_xy[:, 0] = tracker_x[:, -1]
    final_xy[:, 1] = tracker_y[:, -1]
    initial_xy = np.zeros((num_pts, 2))
    initial_xy[:, 0] = tracker_x[:, 0]
    initial_xy[:, 1] = tracker_y[:, 0]
    # find the edge points -- initial tracking points closest to the edge of the wound
    edge_pts = []
    for kk in range(0, wound_contour.shape[0]):
        x = wound_contour[kk, 0]
        y = wound_contour[kk, 1]
        dist = distance.cdist(np.asarray([[x, y]]), initial_xy, 'euclidean')
        argmin = np.argmin(dist)
        edge_pts.append(argmin)
    edge_pts = get_unique(edge_pts)
    # convert the edge points into an alpha shape
    num_pts = len(edge_pts)
    points_2d_initial = []
    points_2d_final = []
    for kk in range(0, num_pts):
        ix = edge_pts[kk]
        points_2d_initial.append((initial_xy[ix, 0], initial_xy[ix, 1]))
        points_2d_final.append((final_xy[ix, 0], final_xy[ix, 1]))
    if alpha_assigned:
        alpha_shape_initial = alphashape.alphashape(points_2d_initial, assigned_alpha)
        alpha_shape_final = alphashape.alphashape(points_2d_final, assigned_alpha)
    else:  # this will automatically select alpha, however it can be slow
        alpha_shape_initial = alphashape.alphashape(points_2d_initial)
        alpha_shape_final = alphashape.alphashape(points_2d_final)
    # convert the alpha shape into wound masks
    mask_wound_initial = np.zeros(frame_0_mask.shape)
    mask_wound_final = np.zeros(frame_0_mask.shape)
    # convert to a mask
    for kk in range(0, frame_0_mask.shape[0]):
        for jj in range(0, frame_0_mask.shape[1]):
            if alpha_shape_initial.contains(Point(jj, kk)) is True:
                mask_wound_initial[kk, jj] = 1
            if alpha_shape_final.contains(Point(jj, kk)) is True:
                mask_wound_final[kk, jj] = 1
    mask_wound_initial = mask_wound_initial > 0
    mask_wound_final = mask_wound_final > 0
    return mask_wound_initial, mask_wound_final


def perform_tracking(frame_0_mask: np.ndarray, img_list: List, include_reverse: bool = True, wound_contour: np.ndarray = None):
    """Given an initial mask and all images. Will perform forward and reverse (optional) tracking."""
    # convert img_list to all uint8 images
    img_list_uint8 = uint16_to_uint8_all(img_list)
    len_img_list = len(img_list_uint8)
    # perform forward tracking
    is_forward = True
    order_list = get_order_track(len_img_list, is_forward)
    tracker_x, tracker_y = track_all_steps(img_list_uint8, frame_0_mask, order_list)
    if include_reverse:
        # create wound mask
        _, frame_final_mask = wound_mask_from_points(frame_0_mask, tracker_x, tracker_y, wound_contour)
        # perform reverse tracking
        is_forward = False
        order_list = get_order_track(len_img_list, is_forward)
        tracker_x_reverse, tracker_y_reverse = track_all_steps(img_list_uint8, frame_final_mask, order_list)
        # reverse array
        tracker_x_reverse = np.flip(tracker_x_reverse, axis=1)
        tracker_y_reverse = np.flip(tracker_y_reverse, axis=1)
    else:
        tracker_x_reverse = None
        tracker_y_reverse = None
    return tracker_x, tracker_y, tracker_x_reverse, tracker_y_reverse


def check_broken_tissue(tissue_mask: np.ndarray) -> bool:
    """Given a tissue mask. Will return true if it's a broken tissue."""
    # fill the tissue mask (should make this work even with a big wound?)
    # radius = 50
    # tissue_mask_filled = close_region(tissue_mask, radius)
    # region_props = get_region_props(tissue_mask_filled)
    region_props = get_region_props(tissue_mask)
    if len(region_props) == 0:
        return True
    largest_region = get_largest_regions(region_props, 1)[0]
    _, _, _, _, _, _, (min_row, min_col, max_row, max_col), _ = extract_region_props(largest_region)
    mid_row = int(min_row * 0.5 + max_row * 0.5)
    mid_col = int(min_col * 0.5 + max_col * 0.5)
    Q1_area = np.sum(tissue_mask[min_row:mid_row, min_col:mid_col])
    Q2_area = np.sum(tissue_mask[mid_row:max_row, min_col:mid_col])
    Q3_area = np.sum(tissue_mask[mid_row:max_row, mid_col:max_col])
    Q4_area = np.sum(tissue_mask[min_row:mid_row, mid_col:max_col])
    Q_list = [Q1_area, Q2_area, Q3_area, Q4_area]
    min_area = np.min(Q_list)
    max_area = np.max(Q_list)
    mean_area = np.mean(Q_list)
    if min_area / max_area < 0.25 or min_area / mean_area < 0.5:
        is_broken = True
    else:
        is_broken = False
    return is_broken


def check_broken_tissue_all(tissue_mask_list: List) -> List:
    """Given a tissue mask list. Will return a list of booleans specifying if tissue is broken."""
    is_broken_list = []
    for tissue_mask in tissue_mask_list:
        is_broken = check_broken_tissue(tissue_mask)
        is_broken_list.append(is_broken)
    return is_broken_list


# def check_wound_closed(tissue_mask: np.ndarray) -> bool:
#     rad_1 = 1  # close single pixel holes
#     rad_2 = 50  # should close a wound
#     rad_3 = 5  # smallest reasonable wound size
#     tissue_mask_close_1 = close_region(tissue_mask, rad_1) * 1.0
#     tissue_mask_close_2 = close_region(tissue_mask_close_1, rad_2) * 1.0
#     # if the difference between these two partially closed masks is large, the wound is open
#     diff = np.abs(tissue_mask_close_1 - tissue_mask_close_2)
#     thresh = np.min([np.pi * rad_3 ** 2.0, np.sum(tissue_mask) / 8.0])
#     if np.sum(diff) > thresh:
#         is_closed = False
#     else:
#         is_closed = True
#     return is_closed


def shrink_bounding_box(min_row: int, min_col: int, max_row: int, max_col: int, shrink_factor: Union[int, float]) -> tuple:
    """Will return a shrunken bounding box."""
    row_range = max_row - min_row
    col_range = max_col - min_col
    row_delta = int(row_range * shrink_factor * 0.5)
    col_delta = int(col_range * shrink_factor * 0.5)
    min_row_new = min_row + row_delta
    max_row_new = max_row - row_delta
    min_col_new = min_col + col_delta
    max_col_new = max_col - col_delta
    return (min_row_new, min_col_new, max_row_new, max_col_new)


def check_inside_box(region: object, bbox1: tuple, bbox2: tuple) -> bool:
    """Will check if a region is inside an admissible bounding box."""
    _, _, _, cr, cc, _, (min_row, min_col, max_row, max_col), _ = extract_region_props(region)
    inside_bbox = (min_row > bbox1[0]) and (min_col > bbox1[1]) and (max_row < bbox1[2]) and (max_col < bbox1[3])
    centroid_inside_bbox = (cr > bbox2[0]) and (cc > bbox2[1]) and (cr < bbox2[2]) and (cc < bbox2[3])
    if inside_bbox and centroid_inside_bbox:
        return True
    else:
        return False


def check_above_min_size(region: object, min_area: Union[int, float]):
    """Will check if region is above a minimum area."""
    wound_area, _, _, _, _, _, _, _ = extract_region_props(region)
    if wound_area > min_area:
        return True
    else:
        return False


def check_wound_closed(tissue_mask: np.ndarray, wound_region: object) -> bool:
    # use the tissue mask to define an admissible wound region
    # check to make sure the wound is within that region
    # check to make sure the wound is above a certain size
    # get tissue mask bounding box
    if wound_region is None:
        return True
    tissue_object = get_region_props(tissue_mask)[0]
    _, _, _, _, _, _, (min_row, min_col, max_row, max_col), _ = extract_region_props(tissue_object)
    # contract the bounding box to include only the admissible wound area
    shrink_factor = 0.25
    bbox_outer = shrink_bounding_box(min_row, min_col, max_row, max_col, shrink_factor)
    shrink_factor = 0.5
    bbox_inner = shrink_bounding_box(min_row, min_col, max_row, max_col, shrink_factor)
    # make checks on the wound
    is_inside_box = check_inside_box(wound_region, bbox_outer, bbox_inner)
    min_area = (tissue_mask.shape[0] / 100) ** 2.0
    is_large_enough = check_above_min_size(wound_region, min_area)
    if is_inside_box and is_large_enough:
        return False
    else:
        return True


def check_wound_closed_all(tissue_mask_list: List, wound_region_list: List) -> List:
    """Given tissue and wound lists. Will return a list if all tissues are closed."""
    check_wound_closed_list = []
    for kk in range(0, len(tissue_mask_list)):
        is_closed = check_wound_closed(tissue_mask_list[kk], wound_region_list[kk])
        check_wound_closed_list.append(is_closed)
    return check_wound_closed_list


def numpy_to_list(input_path: Path, file_name: str) -> List:
    """Given an input directory and a file name. Import all np arrays and return as a list."""
    converted_to_list = []
    file_names = glob.glob(str(input_path) + '/' + file_name + '*.npy')
    for file in file_names:
        array = np.load(file)
        converted_to_list.append(array)
    return converted_to_list


def run_seg_visualize(
    output_path: Path,
    img_list: List,
    contour_list: List,
    tissue_parameters_list: List,
    is_broken_list: List,
    is_closed_list: List,
    fname: str
) -> tuple:
    """Given input and output information. Run segmentation visualization."""
    # path_list = save_all_img_with_contour(output_path, fname, img_list, contour_list, is_broken_list, is_closed_list)
    path_list = save_all_img_with_contour_and_width(output_path, fname, img_list, contour_list, tissue_parameters_list, is_broken_list, is_closed_list)
    gif_path = create_gif(output_path, fname, path_list)
    return (path_list, gif_path)


def run_bf_seg_vs_fl_seg_visualize(
    output_path: Path,
    img_list: List,
    contour_list_bf: List,
    contour_list_fl: List,
) -> tuple:
    """Given input and output information. Run seg comparison visualization."""
    fname = "bf_with_fl"
    path_list = save_all_img_with_double_contour(output_path, fname, img_list, contour_list_bf, contour_list_fl)
    gif_path = create_gif(output_path, fname, path_list)
    return (path_list, gif_path)


def run_all(folder_path: Path) -> List:
    """Given a folder path. Will read input, run code, generate all outputs."""
    time_all = []
    action_all = []
    time_all.append(time.time())
    action_all.append("start")
    input_dict, input_path_dict, output_path_dict = input_info_to_dicts(folder_path)
    time_all.append(time.time())
    action_all.append("loaded input")
    if input_dict["segment_brightfield"] is True:
        input_path = input_path_dict["brightfield_images_path"]
        output_path = output_path_dict["segment_brightfield_path"]
        thresh_fcn = select_threshold_function(input_dict, True, False, False)
        # throw errors here if input_path == None? (future) and/or output dir isn't created
        _, _, _, _, _, _, _, _, _, img_list_bf, contour_list_bf, tissue_param_list_bf, is_broken_list_bf, is_closed_list_bf = run_segment(input_path, output_path, thresh_fcn)
        time_all.append(time.time())
        action_all.append("segmented brightfield")
    if input_dict["segment_fluorescent"] is True:
        input_path = input_path_dict["fluorescent_images_path"]
        output_path = output_path_dict["segment_fluorescent_path"]
        thresh_fcn = select_threshold_function(input_dict, False, True, False)
        # throw errors here if input_path == None? (future) and/or output dir isn't created
        _, _, _, _, _, _, _, _, _, img_list_fl, contour_list_fl, tissue_param_list_fl, is_broken_list_fl, is_closed_list_fl = run_segment(input_path, output_path, thresh_fcn)
        time_all.append(time.time())
        action_all.append("segmented fluorescent")
    if input_dict["segment_ph1"] is True:
        input_path = input_path_dict["ph1_images_path"]
        output_path = output_path_dict["segment_ph1_path"]
        thresh_fcn = select_threshold_function(input_dict, False, False, True)
        # throw errors here if input_path == None? (future) and/or output dir isn't created
        _, _, _, _, _, _, _, _, _, img_list_ph1, contour_list_ph1, tissue_param_list_ph1, is_broken_list_ph1, is_closed_list_ph1 = run_segment(input_path, output_path, thresh_fcn)
        time_all.append(time.time())
        action_all.append("segmented ph1")
    if input_dict["seg_bf_visualize"] is True:
        output_path = output_path_dict["segment_brightfield_vis_path"]
        fname = "brightfield_contour"
        _ = run_seg_visualize(output_path, img_list_bf, contour_list_bf, tissue_param_list_bf, is_broken_list_bf, is_closed_list_bf, fname)
        # throw errors here if necessary segmentation data doesn't exist
        time_all.append(time.time())
        action_all.append("visualized brightfield")
    if input_dict["seg_fl_visualize"] is True:
        output_path = output_path_dict["segment_fluorescent_vis_path"]
        fname = "fluorescent_contour"
        _ = run_seg_visualize(output_path, img_list_fl, contour_list_fl, tissue_param_list_fl, is_broken_list_fl, is_closed_list_fl, fname)
        # throw errors here if necessary segmentation data doesn't exist
        time_all.append(time.time())
        action_all.append("visualized fluorescent")
    if input_dict["seg_ph1_visualize"] is True:
        output_path = output_path_dict["segment_ph1_vis_path"]
        fname = "ph1_contour"
        _ = run_seg_visualize(output_path, img_list_ph1, contour_list_ph1, tissue_param_list_ph1, is_broken_list_ph1, is_closed_list_ph1, fname)
        # throw errors here if necessary segmentation data doesn't exist
        time_all.append(time.time())
        action_all.append("visualized ph1")
    if input_dict["bf_seg_with_fl_seg_visualize"] is True:
        output_path = output_path_dict["bf_seg_with_fl_seg_visualize_path"]
        _ = run_bf_seg_vs_fl_seg_visualize(output_path, img_list_bf, contour_list_bf, contour_list_fl)
        # throw errors here if necessary segmentation data doesn't exist
        time_all.append(time.time())
        action_all.append("visualized brightfield and fluorescent")
    return time_all, action_all
