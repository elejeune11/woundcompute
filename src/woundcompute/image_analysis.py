import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy import ndimage
from skimage import io
from skimage import measure
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from typing import List, Union
import yaml


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
    """Given a list of region properties. Will return a list of the 3 largest regions.
    If there are fewer than 3 regions, will return all regions."""
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
    area = region_props.area
    axis_major_length = region_props.axis_major_length
    axis_minor_length = region_props.axis_minor_length
    centroid = region_props.centroid
    centroid_row = centroid[0]
    centroid_col = centroid[1]
    coords = region_props.coords
    return area, axis_major_length, axis_minor_length, centroid_row, centroid_col, coords


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
    contours = measure.find_contours(blur, 0.95)
    contour_list = []
    contour_leng = []
    for cont in contours:
        contour_list.append(cont)
        contour_leng.append(cont.shape[0])
    argmax = np.argmax(contour_leng)
    chosen_contour = contour_list[argmax]
    return chosen_contour


def mask_to_area(mask: np.ndarray, pix_to_microns: Union[float, int] = 1):
    """Given a mask and pixel to micron conversions. Returns wound area."""
    area = np.sum(mask)
    area_scaled = area * pix_to_microns * pix_to_microns
    return area_scaled


def threshold_gfp_v1(array: np.ndarray) -> np.ndarray:
    """Given a gfp image array. Will return a binary array where gfp = 0, background = 1."""
    median_filter_size = 5
    array_median = apply_median_filter(array, median_filter_size)
    gaussian_filter_size = 1
    array_gaussian = apply_gaussian_filter(array_median, gaussian_filter_size)
    thresh_img = apply_otsu_thresh(array_gaussian)
    thresh_img_inverted = invert_mask(thresh_img)
    return thresh_img_inverted


def threshold_brightfield_v1(array: np.ndarray) -> np.ndarray:
    """Given a brightfield image array. Will return a binary array where tissue = 0, background = 1."""
    median_filter_size = 5
    array_median = apply_median_filter(array, median_filter_size)
    gaussian_filter_size = 2
    array_gaussian = apply_gaussian_filter(array_median, gaussian_filter_size)
    thresh_img = apply_otsu_thresh(array_gaussian)
    return thresh_img


def isolate_masks(array: np.ndarray) -> np.ndarray:
    """Given a binary mask where background = 1. Will return a mask where `tissue' = 1.
    Will return a mask where `wound' = 1."""
    # select the three largest "background" regions -- side, side, wound
    region_props = get_region_props(array)
    num_regions = 3
    regions_largest = get_largest_regions(region_props, num_regions)
    # identify the wound as the "background" region closest to the center
    center_0, center_1 = get_domain_center(array)
    wound_region = get_closest_region(regions_largest, center_0, center_1)
    # create the wound mask
    wound_region_coords = region_to_coords([wound_region])
    wound_mask = coords_to_mask(wound_region_coords, array)
    # create the tissue mask
    regions_largest_coords = region_to_coords(regions_largest)
    tissue_mask_extra = coords_to_inverted_mask(regions_largest_coords, array)
    region_props = get_region_props(tissue_mask_extra)
    num_regions = 1
    regions_largest = get_largest_regions(region_props, num_regions)
    regions_largest_coords = region_to_coords(regions_largest)
    tissue_mask = coords_to_mask(regions_largest_coords, array)
    return tissue_mask, wound_mask, wound_region


def read_tiff(img_path: Path) -> np.ndarray:
    """Given a path to a tiff. Will return an array."""
    img = io.imread(img_path)
    return img


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
    return


def show_and_save_contour(
    img_array: np.ndarray,
    contour: np.ndarray,
    save_path: Path,
    title: str = " "
) -> None:
    """Given an image, contour, and path location. Will plot and save."""
    plt.figure()
    plt.imshow(img_array, cmap=plt.cm.gray)
    plt.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2.0)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    return


def save_numpy(array: np.ndarray, save_path: Path) -> None:
    """Given a numpy array and path location. Will save as numpy array."""
    np.save(save_path, array)
    return


def save_yaml(
    area: Union[float, int],
    axis_major_length: Union[float, int],
    axis_minor_length: Union[float, int],
    centroid_row: Union[float, int],
    centroid_col: Union[float, int],
    yaml_path: Path
) -> None:
    """Given wound properties and yaml path. Will save properties as a yaml file."""
    Dict = {"wound_area": area,
            "axis_major_length": axis_major_length,
            "axis_minor_length": axis_minor_length,
            "centroid_row": centroid_row,
            "centroid_col": centroid_col
            }
    with open(yaml_path, 'w') as outfile:
        yaml.dump(Dict, outfile, default_flow_style=False)
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
            "seg_fl_verison",
            "seg_fl_visualize",
            "track_brightfield",
            "track_bf_version",
            "track_bf_visualize",
            "bf_seg_with_fl_seg_visualize",
            "bf_track_with_fl_seg_visualize",
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
    return path_dict


def input_info_to_dicts(folder_path: Path) -> dict:
    """Given a folder path. Will get input and output dictionaries set up."""
    input_dict = input_info_to_input_dict(folder_path)
    input_path_dict = input_info_to_input_paths(folder_path)
    output_path_dict = input_info_to_output_paths(folder_path, input_dict)
    return input_dict, input_path_dict, output_path_dict


# def (input_path_dict: dict, output_path_dict: dict)

# def analyze_image(
#     img_path: Path,
#     is_brightfield: bool,
#     tissue_mask_path: Path,
#     wound_mask_path: Path,
#     contour_path: Path,
#     yaml_path: Path,
#     vis_path: Path
# ) -> None:
#     """Given an image path. Will run all analysis."""
#     file = read_tiff(img_path)
#     if is_brightfield:
#         file_thresh = threshold_brightfield_v1(file)
#     else:
#         file_thresh = threshold_gfp_v1(file)
#     tissue_mask, wound_mask, wound_region = isolate_masks(file_thresh)
#     contour = mask_to_contour(wound_mask)
#     area, axis_major_length, axis_minor_length, centroid_row, centroid_col, coords = extract_region_props(wound_region)
#     # save numpy arrays
#     save_numpy(tissue_mask, tissue_mask_path)
#     save_numpy(wound_mask, wound_mask_path)
#     save_numpy(contour, contour_path)
#     save_yaml(area, axis_major_length, axis_minor_length, centroid_row, centroid_col, yaml_path)
#     # plot and save visualization
#     show_and_save_contour(file, contour, vis_path)
#     return


# def read_tiff_stack(img_path: Path) -> np.ndarray:

# def analyze_multi_image():
