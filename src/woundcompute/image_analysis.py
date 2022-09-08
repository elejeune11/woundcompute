import glob
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy import ndimage
from skimage import exposure, img_as_ubyte
from skimage import io
from skimage import measure
from skimage.filters import threshold_otsu
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
    else:
        raise ValueError("specified version is not supported")


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
    plt.plot(contour_bf[:, 1], contour_bf[:, 0], 'r', linewidth=2.0)
    plt.plot(contour_fl[:, 1], contour_fl[:, 0], 'c:', linewidth=2.0)
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


def select_threshold_function(input_dict: dict, is_brightfield: bool) -> int:
    """Given setup information. Will return which segmentation function to run."""
    if is_brightfield and input_dict["seg_bf_version"] == 1:
        return 1
    elif is_brightfield is False and input_dict["seg_fl_version"] == 1:
        return 2
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
        save_numpy(array_list[kk], save_path)
        file_name_list.append(save_path)
    return file_name_list


def save_all_img_with_contour(folder_path: Path, file_name: str, img_list: List, contour_list: List) -> List:
    "Given segmentation results. Plot and save image and contour."
    file_name_list = []
    for kk in range(0, len(img_list)):
        img = img_list[kk]
        cont = contour_list[kk]
        save_path = folder_path.joinpath(file_name + "_%05d.png" % (kk)).resolve()
        title = "frame %05d" % (kk)
        show_and_save_contour(img, cont, save_path, title)
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


def mask_all(thresh_img_list: List) -> List:
    """Given a thresholded image list. Will return masks and wound regions."""
    tissue_mask_list = []
    wound_mask_list = []
    wound_region_list = []
    for thresh_img in thresh_img_list:
        tissue_mask, wound_mask, wound_region = isolate_masks(thresh_img)
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


def parameters_all(wound_region_list: List) -> List:
    """Given a wound regions list. Will return wound properties list"""
    area_list = []
    axis_major_length_list = []
    axis_minor_length_list = []
    for wound_region in wound_region_list:
        area, axis_major_length, axis_minor_length, _, _, _ = extract_region_props(wound_region)
        area_list.append(area)
        axis_major_length_list.append(axis_major_length)
        axis_minor_length_list.append(axis_minor_length)
    return area_list, axis_major_length_list, axis_minor_length_list


def run_segment(input_path: Path, output_path: Path, threshold_function_idx: int) -> List:
    """Given input and output information. Will run the complete segmentation process."""
    # read the inputs
    img_list = read_all_tiff(input_path)
    # apply threshold
    thresholded_list = threshold_all(img_list, threshold_function_idx)
    # masking
    tissue_mask_list, wound_mask_list, wound_region_list = mask_all(thresholded_list)
    # contour
    contour_list = contour_all(wound_mask_list)
    # parameters
    area_list, axis_major_length_list, axis_minor_length_list = parameters_all(wound_region_list)
    # save numpy arrays
    wound_name_list = save_all_numpy(output_path, "wound_mask", wound_mask_list)
    tissue_name_list = save_all_numpy(output_path, "tissue_mask", tissue_mask_list)
    contour_name_list = save_all_numpy(output_path, "contour_coords", contour_list)
    # save lists
    area_path = save_list(output_path, "wound_area_vs_frame", area_list)
    ax_maj_path = save_list(output_path, "wound_major_axis_length_vs_frame", axis_major_length_list)
    ax_min_path = save_list(output_path, "wound_minor_axis_length_vs_frame", axis_minor_length_list)
    return wound_name_list, tissue_name_list, contour_name_list, area_path, ax_maj_path, ax_min_path, img_list, contour_list


def numpy_to_list(input_path: Path, file_name: str) -> List:
    """Given an input directory and a file name. Import all np arrays and return as a list."""
    converted_to_list = []
    file_names = glob.glob(str(input_path) + '/' + file_name + '*.npy')
    for file in file_names:
        array = np.load(file)
        converted_to_list.append(array)
    return converted_to_list


def run_seg_visualize(output_path: Path, img_list: List, contour_list: List, fname: str) -> tuple:
    """Given input and output information. Run segmentation visualization."""
    path_list = save_all_img_with_contour(output_path, fname, img_list, contour_list)
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
        thresh_fcn = select_threshold_function(input_dict, True)
        # throw errors here if input_path == None? (future) and/or output dir isn't created
        _, _, _, _, _, _, img_list_bf, contour_list_bf = run_segment(input_path, output_path, thresh_fcn)
        time_all.append(time.time())
        action_all.append("segmented brightfield")
    if input_dict["segment_fluorescent"] is True:
        input_path = input_path_dict["fluorescent_images_path"]
        output_path = output_path_dict["segment_fluorescent_path"]
        thresh_fcn = select_threshold_function(input_dict, False)
        # throw errors here if input_path == None? (future) and/or output dir isn't created
        _, _, _, _, _, _, img_list_fl, contour_list_fl = run_segment(input_path, output_path, thresh_fcn)
        time_all.append(time.time())
        action_all.append("segmented fluorescent")
    if input_dict["seg_bf_visualize"] is True:
        output_path = output_path_dict["segment_brightfield_vis_path"]
        fname = "brightfield_contour"
        _ = run_seg_visualize(output_path, img_list_bf, contour_list_bf, fname)
        # throw errors here if necessary segmentation data doesn't exist
        time_all.append(time.time())
        action_all.append("visualized brightfield")
    if input_dict["seg_fl_visualize"] is True:
        output_path = output_path_dict["segment_fluorescent_vis_path"]
        fname = "fluorescent_contour"
        _ = run_seg_visualize(output_path, img_list_fl, contour_list_fl, fname)
        # throw errors here if necessary segmentation data doesn't exist
        time_all.append(time.time())
        action_all.append("visualized fluorescent")
    if input_dict["bf_seg_with_fl_seg_visualize"] is True:
        output_path = output_path_dict["bf_seg_with_fl_seg_visualize_path"]
        _ = run_bf_seg_vs_fl_seg_visualize(output_path, img_list_bf, contour_list_bf, contour_list_fl)
        # throw errors here if necessary segmentation data doesn't exist
        time_all.append(time.time())
        action_all.append("visualized brightfield and fluorescent")
    return time_all, action_all
