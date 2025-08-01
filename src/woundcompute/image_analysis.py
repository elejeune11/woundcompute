import glob
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from skimage import io
from scipy.ndimage import shift
import time
from typing import List,Tuple
import yaml
from PIL import Image
from skimage.transform import rescale
import re
from woundcompute import segmentation as seg
from woundcompute import compute_values as com
from woundcompute import texture_tracking as tt
from woundcompute import post_process as pp


def hello_wound_compute() -> str:
    "Given no input. Simple hello world as a test function."
    return "Hello World!"


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
    frame_num: int = None,
    title: str = " ",
    broken_frame: int = None,
    closed_frame: int = None,
    pillars_pos_x: List = None,
    pillars_pos_y: List = None,
    pillar_contours: List = None,
) -> None:
    """Given an image, contour, and path location. Will plot and save."""

    img_h,img_w = img_array.shape
    xt_broken = 2.6 * img_w / 8.0
    yt_broken = 0.8 * img_h / 8.0
    xt_closed = 2.4 * img_w / 8.0
    yt_closed = 7.5 * img_h / 8.0

    if img_h > 512 or img_h > 512:
        scale_factor = 1 / ( (img_w/512 + img_h/512) / 2 )
        img_array = rescale(img_array,scale_factor,anti_aliasing=True)
        contour=contour*scale_factor if contour is not None else None
        points=np.array(points)*scale_factor if points is not None else None
        pillars_pos_x=pillars_pos_x*scale_factor if pillars_pos_x is not None else None
        pillars_pos_y=pillars_pos_y*scale_factor if pillars_pos_y is not None else None
        if pillar_contours is not None:
            pillar_contours=[pil_cont*scale_factor for pil_cont in pillar_contours]
        xt_broken=xt_broken*scale_factor
        yt_broken=yt_broken*scale_factor
        xt_closed=xt_closed*scale_factor
        yt_closed=yt_closed*scale_factor
    else:
        scale_factor = 1

    plt.figure()
    plt.imshow(img_array, cmap=plt.cm.gray)


    if points is not None:
        plt.plot(points[1], points[0], 'k-o', linewidth=2.0, antialiased=True)
    if is_broken or broken_frame:
        if broken_frame is None:
            broken_frame = frame_num
        plt.text(xt_broken, yt_broken, f"broken at frame {broken_frame}", color="r", backgroundcolor="w", fontsize=17)
    if is_closed or closed_frame:
        if closed_frame is None:
            closed_frame = frame_num
        plt.text(xt_closed, yt_closed, f"closed at frame {closed_frame}", color="r", backgroundcolor="w", fontsize=17)
    if contour is not None:
        plt.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2.0, antialiased=True)
    if pillars_pos_x is not None and pillars_pos_y is not None:
        dot_size = 0.00003 * img_w * img_h * scale_factor
        plt.scatter(pillars_pos_x,pillars_pos_y,s=dot_size,c='blue')
    if pillar_contours is not None:
        for pil_ind,pil_cont in enumerate(pillar_contours):
            contour_x_mean = np.mean(pil_cont[:,0])
            contour_y_mean = np.mean(pil_cont[:,1])
            pil_cont[:,0] -= contour_x_mean
            pil_cont[:,1] -= contour_y_mean

            pil_cont[:,0] = pil_cont[:,0] + pillars_pos_x[pil_ind]
            pil_cont[:,1] = pil_cont[:,1] + pillars_pos_y[pil_ind]
            plt.plot(pil_cont[:,0],pil_cont[:,1],'blue',linewidth=2.0, antialiased=True)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return broken_frame,closed_frame


def show_and_save_bi_tissue(
    img_array: np.ndarray,
    is_broken: bool,
    save_path: Path,
    frame_num: int = None,
    title: str = " ",
    broken_frame: int = None,
    pillars_pos_x: List = None,
    pillars_pos_y: List = None,
    pillar_contours: List = None,
) -> None:
    """Given an image, contour, and path location. Will plot and save.
    Currently, this function is a placeholder for saving information related to before injury tissues.
    Information will be included per request of users later.
    """

    img_h,img_w = img_array.shape
    xt_broken = 2.6 * img_w / 8.0
    yt_broken = 0.8 * img_h / 8.0
    xt_closed = 2.4 * img_w / 8.0
    yt_closed = 7.5 * img_h / 8.0

    if img_h > 512 or img_h > 512:
        scale_factor = 1 / ( (img_w/512 + img_h/512) / 2 )
        img_array = rescale(img_array,scale_factor,anti_aliasing=True)
        pillars_pos_x=pillars_pos_x*scale_factor if pillars_pos_x is not None else None
        pillars_pos_y=pillars_pos_y*scale_factor if pillars_pos_y is not None else None
        if pillar_contours is not None:
            pillar_contours=[pil_cont*scale_factor for pil_cont in pillar_contours]
        xt_broken=xt_broken*scale_factor
        yt_broken=yt_broken*scale_factor
        xt_closed=xt_closed*scale_factor
        yt_closed=yt_closed*scale_factor
    else:
        scale_factor = 1

    plt.figure()
    plt.imshow(img_array, cmap=plt.cm.gray)

    if is_broken or broken_frame:
        if broken_frame is None:
            broken_frame = frame_num
        plt.text(xt_broken, yt_broken, f"broken at frame {broken_frame}", color="r", backgroundcolor="w", fontsize=17)
    if pillars_pos_x is not None and pillars_pos_y is not None:
        dot_size = 0.00003 * img_w * img_h * scale_factor
        plt.scatter(pillars_pos_x,pillars_pos_y,s=dot_size,c='blue')
    if pillar_contours is not None:
        for pil_ind,pil_cont in enumerate(pillar_contours):
            contour_x_mean = np.mean(pil_cont[:,0])
            contour_y_mean = np.mean(pil_cont[:,1])
            pil_cont[:,0] -= contour_x_mean
            pil_cont[:,1] -= contour_y_mean

            pil_cont[:,0] = pil_cont[:,0] + pillars_pos_x[pil_ind]
            pil_cont[:,1] = pil_cont[:,1] + pillars_pos_y[pil_ind]
            plt.plot(pil_cont[:,0],pil_cont[:,1],'blue',linewidth=2.0, antialiased=True)

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
            "zoom_type",
            "track_pillars_ph1"
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
    if len(name_list) == 0:
        name_list = glob.glob(str(folder_path) + '/*.tif')
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
    if input_dict["track_pillars_ph1"] is True:
        track_pillars_ph1_path = create_folder(folder_path, "track_pillars_ph1")
        path_dict["track_pillars_ph1_path"] = track_pillars_ph1_path
    else:
        path_dict["track_pillars_ph1_path"] = None
    return path_dict


def input_info_to_dicts(folder_path: Path) -> dict:
    """Given a folder path. Will get input and output dictionaries set up."""
    input_dict = input_info_to_input_dict(folder_path)
    input_path_dict = input_info_to_input_paths(folder_path)
    output_path_dict = input_info_to_output_paths(folder_path, input_dict)
    return input_dict, input_path_dict, output_path_dict


def read_all_tiff(folder_path: Path) -> List:
    """Given a folder path. Will return a list of all tiffs as an array."""
    path_list = image_folder_to_path_list(folder_path)
    tiff_list = []
    for path in path_list:
        array = read_tiff(path)
        tiff_list.append(array)
    return tiff_list


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
    is_closed_list: List,
    avg_pos_all_x: List = None,
    avg_pos_all_y: List = None,
    pillar_masks: List = None,
    is_in_bi_folder: bool = False,
) -> List:
    "Given segmentation results. Plot and save image and contour."
    file_name_list = []
    broken_frame = None
    closed_frame = None

    if pillar_masks is not None:
        pillar_contours_list = seg.contour_all(pillar_masks)
    else:
        pillar_contours_list = None
    
    for kk in range(0, len(img_list)):
        img = img_list[kk]

        if avg_pos_all_x is None:
            pillars_pos_x = None
            pillars_pos_y = None
        else:
            pillars_pos_x = avg_pos_all_x[kk]
            pillars_pos_y = avg_pos_all_y[kk]

        if not is_in_bi_folder:
            cont = contour_list[kk]
            is_broken = is_broken_list[kk]
            is_closed = is_closed_list[kk]
            
            #  area, pt1_0, pt1_1, pt2_0, pt2_1, width, kappa_1, kappa_2
            tp = tissue_parameters_list[kk]
            points = [[tp[1], tp[3]], [tp[2], tp[4]]]
            save_path = folder_path.joinpath(file_name + "_%05d.png" % (kk)).resolve()
            title = "frame %05d" % (kk)
            broken_frame,closed_frame=show_and_save_contour_and_width(img, cont, is_broken, is_closed, points, save_path, title=title,
                                            frame_num = kk,broken_frame=broken_frame,closed_frame=closed_frame,
                                            pillars_pos_x=pillars_pos_x,pillars_pos_y=pillars_pos_y,pillar_contours=pillar_contours_list)
        elif is_in_bi_folder:
            save_path = folder_path.joinpath(file_name + "_%05d.png" % (kk)).resolve()
            show_and_save_bi_tissue(img_array=img,is_broken=is_broken_list[kk],save_path=save_path,frame_num=kk,
                                    title=f"before injury frame {kk}",pillars_pos_x=pillars_pos_x,pillars_pos_y=pillars_pos_y,pillar_contours=pillar_contours_list)
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

# def line_param(
#     centroid_row: Union[float, int],
#     centroid_col: Union[float, int],
#     orientation: Union[float, int]
# ) -> Union[float, int]:
#     """Given a point and a slope (orientation). Will return line format as ax_0 + bx_1 + c = 0."""
#     line_a = -1.0 * np.tan(orientation)
#     line_b = 1.0
#     line_c = -1.0 * centroid_row + np.tan(orientation) * centroid_col
#     return line_a, line_b, line_c

# def dist_to_line(
#     line_a: Union[float, int],
#     line_b: Union[float, int],
#     line_c: Union[float, int],
#     pt_0: Union[float, int],
#     pt_1: Union[float, int]
# ) -> Union[float, int]:
#     """Given line parameters and a point. Will return the distance to the line."""
#     numer = np.abs(line_a * pt_0 + line_b * pt_1 + line_c)
#     denom = ((line_a) ** 2.0 + (line_b) ** 2.0) ** 0.5
#     line_dist = numer / denom
#     return line_dist

# def move_point(
#     pt_0: Union[float, int],
#     pt_1: Union[float, int],
#     line_a: Union[float, int],
#     line_b: Union[float, int],
#     line_c: Union[float, int],
#     cutoff: Union[float, int]
# ) -> Union[float, int]:
#     """Given a point and a line. Will move the point to the cutoff."""
#     line_dist = dist_to_line(line_a, line_b, line_c, pt_0, pt_1)
#     if np.abs(line_dist) < 10 ** -6:
#         return pt_0, pt_1
#     if line_a == 0:
#         sig = pt_1 - (line_c / line_b)
#         unit_vec_0 = -1.0 * np.sign(sig)
#         unit_vec_1 = 0
#     elif line_b == 0:
#         unit_vec_0 = 0
#         sig = pt_0 - (line_c / line_a)
#         unit_vec_1 = -1.0 * np.sign(sig)
#     else:
#         # line_0_numer = -1.0 * line_b / line_a * pt_0 + pt_1 + line_c / line_b
#         # line_0_denom = -1.0 * (line_b / line_a + line_a / line_b)
#         line_0_numer = line_c / line_b + pt_1 - line_b / line_a * pt_0
#         line_0_denom = -1.0 * (line_a / line_b + line_b / line_a)
#         line_0 = line_0_numer / line_0_denom
#         line_1 = -1.0 * line_a / line_b * line_0 - line_c / line_b
#         vec_0 = line_0 - pt_0
#         vec_1 = line_1 - pt_1
#         unit_vec_0 = vec_0 / line_dist
#         unit_vec_1 = vec_1 / line_dist
#     pt_0_mod = pt_0 + unit_vec_0 * (line_dist - cutoff)
#     pt_1_mod = pt_1 + unit_vec_1 * (line_dist - cutoff)
#     return pt_0_mod, pt_1_mod

# def clip_contour(
#     contour: np.ndarray,
#     centroid_row: Union[int, float],
#     centroid_col: Union[int, float],
#     orientation: Union[int, float],
#     tissue_axis_major_length: Union[int, float],
#     tissue_axis_minor_length: Union[int, float]
# ) -> np.ndarray:
#     cutoff = tissue_axis_major_length / 3.0
#     line_a, line_b, line_c = line_param(centroid_row, centroid_col, orientation)
#     contour_clipped = []
#     for kk in range(0, contour.shape[0]):
#         pt_0 = contour[kk, 0]
#         pt_1 = contour[kk, 1]
#         line_dist = dist_to_line(line_a, line_b, line_c, pt_1, pt_0)
#         if line_dist < cutoff:
#             contour_clipped.append([pt_0, pt_1])
#         else:
#             pt_1_mod, pt_0_mod = move_point(pt_1, pt_0, line_a, line_b, line_c, cutoff)
#             contour_clipped.append([pt_0_mod, pt_1_mod])
#     # if len(contour_clipped) < contour.shape[0]:
#     #     contour_clipped.append(contour_clipped[0])
#     contour_clipped = np.asarray(contour_clipped)
#     return contour_clipped

# def resample_contour(contour: np.ndarray) -> np.ndarray:
#     """Given a contour. Will resample and return the resampled contour."""
#     ix_0 = contour[:, 0]
#     ix_1 = contour[:, 1]
#     tck, u = splprep([ix_0, ix_1], s=0)
#     resampled_contour_list = splev(u, tck)
#     resampled_contour = np.asarray(resampled_contour_list).T
#     num_pts_max = 250
#     num = np.max([int(resampled_contour.shape[0] / num_pts_max), 1])
#     downsampled_contour = resampled_contour[::num, :]
#     return downsampled_contour

# def get_penalized(contour: np.ndarray, contour_clipped: np.ndarray):
#     """Given the original contour and the clipped contour. Will return penalized contour."""
#     cc_penalized = []
#     for kk in range(0, contour.shape[0]):
#         if math.isclose(contour[kk, 0], contour_clipped[kk, 0]) and math.isclose(contour[kk, 1], contour_clipped[kk, 1]):
#             cc_penalized.append([contour[kk, 0], contour[kk, 1]])
#         else:
#             cc_penalized.append([math.inf, math.inf])
#     cc_penalized = np.asarray(cc_penalized)
#     return cc_penalized

# def get_contouor_distance_across_all_v2(contour: np.ndarray) -> np.ndarray:
#     """Given a contour. Will compute the distance across the contour at every point."""
#     return distance_all, ix_all

# def get_contour_width_v2(contour: np.ndarray) -> Union[float, int]:
#     """Given a contour. Will compute minimum distance across and location of minimum. This is the width."""
#     return width, idx_a, idx_b

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


def subtract_moving_pillar_from_tissue(
        tissue_masks:np.ndarray,
        pillar_mask:np.ndarray,
        pillar_centers:List
    )->np.ndarray:
    """
    Subtracts a moving pillar mask from each frame of tissue masks.

    Parameters:
    - tissue_masks: np.ndarray of shape (T, H, W)
    - pillar_mask: np.ndarray of shape (H, W)
    - pillar_centers: list of (y, x) coordinates, one per frame

    Returns:
    - result_masks: np.ndarray of shape (T, H, W) after subtraction
    """
    T, H, W = tissue_masks.shape
    result_masks = np.copy(tissue_masks)
    
    # Center of the original pillar_mask
    pillar_center_yx = np.array(np.array(np.nonzero(pillar_mask)).mean(axis=1))
    
    for t in range(T):
        # Compute the shift from original center to desired center
        target_center = np.array(pillar_centers[t])
        shift_vec = target_center - pillar_center_yx

        # Shift the pillar mask using scipy.ndimage.shift
        shifted_pillar = shift(pillar_mask.astype(float), shift=shift_vec, order=0, mode='constant', cval=0)
        shifted_pillar = (shifted_pillar > 0.5).astype(np.uint8)

        # Subtract shifted pillar from tissue mask
        result_masks[t] = np.logical_and(result_masks[t], np.logical_not(shifted_pillar)).astype(np.uint8)

    return result_masks


def subtract_moving_pillars_from_tissue_masks(
        tissue_masks:List[np.ndarray],
        pillar_masks:List[np.ndarray],
        pillar_centers_x:np.ndarray,
        pillar_centers_y:np.ndarray
    )-> List[np.ndarray]:
    """
    Subtracts moving pillars from tissue masks.
    Parameters:
    - tissue_masks: np.ndarray of shape (F, H, W)
    - pillar_masks: List of np.ndarray, each of shape (H, W)
    - pillar_centers_x: np.ndarray of shape (F,P), where F is the number of frames and P is the number of pillars
    - pillar_centers_y: np.ndarray of shape (F,P), where F is the number of frames and P is the number of pillars
    Returns:
    - result_masks: np.ndarray of shape (T, H, W) after subtraction
    """

    if len(pillar_centers_x.shape) == 1:
        pillar_centers_x = pillar_centers_x.reshape(1,-1)
        pillar_centers_y = pillar_centers_y.reshape(1,-1)

    tissue_masks_arr = np.array(tissue_masks)
    num_pillars = len(pillar_masks)
    for p in range(num_pillars):
        pillar_mask = pillar_masks[p]
        pillar_centers = list(zip(pillar_centers_y[:, p], pillar_centers_x[:, p]))
        tissue_masks_arr = subtract_moving_pillar_from_tissue(tissue_masks_arr, pillar_mask, pillar_centers)

    # Convert back to list of arrays
    tissue_masks_list = [tissue_masks_arr[i] for i in range(tissue_masks_arr.shape[0])]
    return tissue_masks_list


def run_segment(
    input_path: Path,
    output_path: Path,
    threshold_function_idx: int,
    zoom_fcn_idx: int,
    pillar_pos_x: np.ndarray = None,
    pillar_pos_y: np.ndarray = None,
    is_bi: bool=False) -> List:
    """Given input and output information. Will run the complete segmentation process."""

    if is_bi:
        wound_name_list, tissue_name_list, contour_name_list, area_path, ax_maj_path, ax_min_path, tissue_path, is_broken_path, is_closed_path, img_list, contour_list, tissue_parameters_list, is_broken_list, is_closed_list = run_segment_bi(
            input_path, output_path, threshold_function_idx, zoom_fcn_idx, pillar_pos_x, pillar_pos_y
            )
        return wound_name_list, tissue_name_list, contour_name_list, area_path, ax_maj_path, ax_min_path, tissue_path, is_broken_path, is_closed_path, img_list, contour_list, tissue_parameters_list, is_broken_list, is_closed_list

    # read the inputs
    img_list = read_all_tiff(input_path)
    # apply threshold
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    # masking
    if zoom_fcn_idx == 2:
        # get pillar masks
        # future idea: do this based on multiple images e.g., avg all images?
        selection_idx = 4
        pillar_mask_list = seg.get_pillar_mask_list(img_list[0], selection_idx)
        # do zoom function type 2
        tissue_mask_list, wound_mask_list, wound_region_list = seg.mask_all_with_pillars(thresholded_list, pillar_mask_list)
    else:
        tissue_mask_list, wound_mask_list, wound_region_list = seg.mask_all(thresholded_list, threshold_function_idx)
        pillar_mask_list=None
    # contour
    contour_list = seg.contour_all(wound_mask_list)
    wound_mask_list = seg.contour_to_mask_all(img_list[0],contour_list)
    # wound parameters
    area_list, axis_major_length_list, axis_minor_length_list = com.wound_parameters_all(img_list[0], contour_list)
    wound_area_smoothed_GPR = pp.smooth_with_GPR(np.array(area_list))

    if pillar_mask_list:
        # check if the tissue is broken
        is_broken_list = com.check_broken_tissue_all(
            tissue_mask_list, wound_mask_list, True, zoom_fcn_idx,pillar_mask_list=pillar_mask_list)
        _ = save_all_numpy(output_path, "pillar", pillar_mask_list)

        # refine tissue masks by subtracting moving pillars
        if pillar_pos_x is not None and pillar_pos_y is not None:
            tissue_mask_list = subtract_moving_pillars_from_tissue_masks(
                tissue_mask_list, pillar_mask_list, 
                pillar_pos_x, 
                pillar_pos_y
            )
    else:
        is_broken_list = com.check_broken_tissue_all(
            tissue_mask_list, wound_mask_list, True, zoom_fcn_idx)
    
    # tissue parameters
    tissue_parameters_list = com.tissue_parameters_all(tissue_mask_list, wound_mask_list, zoom_fcn_idx)
    # save numpy arrays
    wound_name_list = save_all_numpy(output_path, "wound_mask", wound_mask_list)
    tissue_name_list = save_all_numpy(output_path, "tissue_mask", tissue_mask_list)
    contour_name_list = save_all_numpy(output_path, "contour_coords", contour_list)
    # save lists
    area_path = save_list(output_path, "wound_area_vs_frame", area_list)
    smoothed_area_path = save_list(output_path,"wound_area_vs_frame_GPR",wound_area_smoothed_GPR)
    smoothed_area_path_png = save_list(output_path, "wound_area_vs_frame", wound_area_smoothed_GPR)
    ax_maj_path = save_list(output_path, "wound_major_axis_length_vs_frame", axis_major_length_list)
    ax_min_path = save_list(output_path, "wound_minor_axis_length_vs_frame", axis_minor_length_list)
    tissue_path = save_list(output_path, "tissue_parameters_vs_frame", tissue_parameters_list)
    is_broken_path = save_list(output_path, "is_broken_vs_frame", is_broken_list)
    # save pngs (plots)
    show_and_save_wound_area(np.array(area_list),wound_area_smoothed_GPR,output_path)
    # check if the wound is closed
    is_closed_list = com.check_wound_closed_all(tissue_mask_list, wound_region_list, zoom_fcn_idx)
    is_closed_path = save_list(output_path, "is_closed_vs_frame", is_closed_list)
    return wound_name_list, tissue_name_list, contour_name_list, area_path, ax_maj_path, ax_min_path, tissue_path, is_broken_path, is_closed_path, img_list, contour_list, tissue_parameters_list, is_broken_list, is_closed_list


def run_segment_bi(
    input_path: Path,
    output_path: Path,
    threshold_function_idx: int,
    zoom_fcn_idx: int,
    pillar_pos_x: np.ndarray=None,
    pillar_pos_y: np.ndarray=None) -> List:
    """Function to run segmentation for before injury (bi) microtissues."""
    # read the inputs
    img_list = read_all_tiff(input_path)
    # apply threshold
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    # masking
    if zoom_fcn_idx == 2:
        # get pillar masks
        # future idea: do this based on multiple images e.g., avg all images?
        selection_idx = 4
        pillar_mask_list = seg.get_pillar_mask_list(img_list[0], selection_idx)
        # do zoom function type 2
        tissue_mask_list, _, _ = seg.mask_all_with_pillars(thresholded_list, pillar_mask_list)
    else:
        tissue_mask_list, _, _ = seg.mask_all(thresholded_list, threshold_function_idx)
        pillar_mask_list=None
    # there are no wound masks for before injury
    wound_mask_list = []
    is_closed_list = []
    for cur_img in img_list:
        wound_mask_list.append(np.zeros_like(cur_img, dtype=np.uint8))
        is_closed_list.append(0)

    if pillar_mask_list:
        # check if the tissue is broken
        is_broken_list = com.check_broken_tissue_all(
            tissue_mask_list, wound_mask_list, True, zoom_fcn_idx,pillar_mask_list=pillar_mask_list)
        _ = save_all_numpy(output_path, "pillar", pillar_mask_list)

        # refine tissue masks by subtracting moving pillars
        if pillar_pos_x is not None and pillar_pos_y is not None:
            tissue_mask_list = subtract_moving_pillars_from_tissue_masks(
                tissue_mask_list, pillar_mask_list, 
                pillar_pos_x, 
                pillar_pos_y
            )
    else:
        is_broken_list = com.check_broken_tissue_all(
            tissue_mask_list, wound_mask_list, True, zoom_fcn_idx)

    # tissue parameters
    tissue_parameters_list = com.tissue_parameters_all(tissue_mask_list, wound_mask_list, zoom_fcn_idx)
    # save numpy arrays
    tissue_name_list = save_all_numpy(output_path, "tissue_mask", tissue_mask_list)
    # save lists
    tissue_path = save_list(output_path, "tissue_parameters_vs_frame", tissue_parameters_list)
    is_broken_path = save_list(output_path, "is_broken_vs_frame", is_broken_list)
    # since the tissue is not injured, we will not have any wound data
    wound_name_list=contour_name_list=area_path=ax_maj_path=ax_min_path=is_closed_path=contour_list=None
    return wound_name_list, tissue_name_list, contour_name_list, area_path, ax_maj_path, ax_min_path, tissue_path, is_broken_path, is_closed_path, img_list, contour_list, tissue_parameters_list, is_broken_list, is_closed_list


def show_and_save_wound_area(
    wound_area:np.ndarray,
    wound_area_smoothed_GPR:np.ndarray,
    output_path:Path,
):
    """
    Given relative distances and GPR smoothed distances, will plot and save the figure.
    Args:
        relative_distances (np.ndarray): Array of relative distances between pillars for each frame.
        GPR_relative_distances (np.ndarray): GPR smoothed relative distances.
        rel_dist_pair_names (np.ndarray): Names of the pillar pairs for which distances are computed.
        output_path (Path): Path to save the figure.
    Returns:
        None
    """
    num_frames = len(wound_area)
    num_pt_GPR = len(wound_area_smoothed_GPR)

    frame_steps = np.linspace(0,num_frames-1,num_frames,dtype=int)
    GPR_steps = np.linspace(0,num_pt_GPR-1,num_pt_GPR,dtype=int)

    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,4))

    ax.scatter(frame_steps,wound_area,c='black',s=8,label='wound area')
    ax.plot(GPR_steps,wound_area_smoothed_GPR,c='red',linewidth=1,label='GPR smoothed')
    ax.set_title('Wound area over time')
    ax.set_xlabel('frame number')
    ax.set_ylabel('wound area (pixels)')
    ax.grid('on',ls=':')

    plt.tight_layout()
    plt.savefig(output_path.joinpath("wound_area.png").resolve())
    return

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


# def check_wound_closed_rotation(tissue_mask_robust: np.ndarray, wound_region: object):
#     # check if no wound
#     if wound_region is None:
#         return True
#     # check if wound is below a certain size
#     min_area = (tissue_mask_robust.shape[0] / 100) ** 2.0
#     is_large_enough = seg.check_above_min_size(wound_region, min_area)
#     if is_large_enough is False:
#         return True
#     # perform rotation, check if wound is inside rotated box
#     _, _, _, _, _, coords, _, _ = seg.extract_region_props(wound_region)
#     center_row, center_col, rot_mat, ang, vec = com.get_rotation_info(center_row_input=None, center_col_input=None, vec_input=None, mask=tissue_mask_robust)
#     rot_mask = com.rot_image(tissue_mask_robust, center_row, center_col, ang)
#     mask_box = com.mask_to_box(rot_mask)
#     row_pts = coords[:, 0]
#     col_pts = coords[:, 1]
#     new_row_pts, new_col_pts = com.rotate_points(row_pts, col_pts, rot_mat, center_row, center_col)


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
    fname: str,
    avg_pos_all_x: List = None,
    avg_pos_all_y: List = None,
    pillar_masks: List = None,
    is_in_bi_folder: bool = False,
) -> tuple:
    """Given input and output information. Run segmentation visualization."""
    # path_list = save_all_img_with_contour(output_path, fname, img_list, contour_list, is_broken_list, is_closed_list)
    path_list = save_all_img_with_contour_and_width(output_path, fname, img_list, contour_list, tissue_parameters_list, is_broken_list, is_closed_list,avg_pos_all_x,avg_pos_all_y,pillar_masks,is_in_bi_folder)
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


def run_texture_tracking(input_path: Path, output_path: Path, threshold_function_idx: int):
    """Given input and output information. Will run texture tracking."""
    # read all tiff images from input path
    img_list = read_all_tiff(input_path)
    # segment the first image to get a frame 0 mask
    img_list_first = [img_list[0]]
    threshold_list = seg.threshold_all(img_list_first, threshold_function_idx)
    # segmenment wound contouor
    tissue_mask_list, wound_mask_list, _ = seg.mask_all(threshold_list, threshold_function_idx)
    frame_0_mask = tissue_mask_list[0]
    wound_mask = wound_mask_list[0]
    wound_contour = seg.mask_to_contour(wound_mask)
    # include reverse tracking
    include_reverse = True
    frame_final_mask, tracker_x_forward, tracker_y_forward, tracker_x_reverse_forward, tracker_y_reverse_forward, wound_area_list, wound_masks_all = tt.perform_tracking(frame_0_mask, img_list, include_reverse, wound_contour)
    # save tracking results
    path_final_frame_mask = output_path.joinpath("tracker_final_wound_mask.txt").resolve()
    np.savetxt(str(path_final_frame_mask), frame_final_mask, fmt="%i")  # TODO: make this more specific -- will be used for key output metrics
    path_tx = output_path.joinpath("tracker_x_forward.txt").resolve()
    np.savetxt(str(path_tx), tracker_x_forward)
    path_ty = output_path.joinpath("tracker_y_forward.txt").resolve()
    np.savetxt(str(path_ty), tracker_y_forward)
    path_txr = output_path.joinpath("tracker_x_reverse_forward.txt").resolve()
    np.savetxt(str(path_txr), tracker_x_reverse_forward)
    path_tyr = output_path.joinpath("tracker_y_reverse_forward.txt").resolve()
    np.savetxt(str(path_tyr), tracker_y_reverse_forward)
    path_wound_area = output_path.joinpath("tracker_wound_area.txt").resolve()
    np.savetxt(str(path_wound_area), np.asarray(wound_area_list))
    path_wound_masks = output_path.joinpath("tracker_wound_masks.npy").resolve()
    np.save(str(path_wound_masks), wound_masks_all)
    return tracker_x_forward, tracker_y_forward, tracker_x_reverse_forward, tracker_y_reverse_forward, wound_area_list, wound_masks_all, path_tx, path_ty, path_txr, path_tyr, path_wound_area, path_wound_masks


def get_subplot_dims(n_plots:int) -> Tuple[int,int]:
    """
    Calculate optimal (rows, cols) for subplots to be as square as possible.
    
    Args:
        n_plots (int): Number of plots needed
        
    Returns:
        tuple: (rows, cols)
    """
    if n_plots == 0:
        return (0, 0)
    
    # Start with square root as first approximation
    sqrt_n = np.sqrt(n_plots)
    rows = round(sqrt_n)
    cols = np.ceil(n_plots / rows)
    
    # Check if we can get a more square arrangement
    # by trying rows ±1 from our initial guess
    candidates = []
    for r in [rows - 1, rows, rows + 1]:
        if r > 0:
            c = np.ceil(n_plots / r)
            aspect_ratio = max(r, c) / min(r, c)
            candidates.append((aspect_ratio, r, c))
    
    # Select the arrangement with the most square aspect ratio
    candidates.sort()
    nrows = int(candidates[0][1])
    ncols = int(candidates[0][2])
    return (nrows, ncols)


def show_and_save_relative_pillar_distances(
    relative_distances:np.ndarray,
    GPR_relative_distances:np.ndarray,
    rel_dist_pair_names:np.ndarray,
    output_path:Path,
    is_potential_bg_shift:bool=False,
):
    """
    Given relative distances and GPR smoothed distances, will plot and save the figure.
    Args:
        relative_distances (np.ndarray): Array of relative distances between pillars for each frame.
        GPR_relative_distances (np.ndarray): GPR smoothed relative distances.
        rel_dist_pair_names (np.ndarray): Names of the pillar pairs for which distances are computed.
        output_path (Path): Path to save the figure.
    Returns:
        None
    """
    num_frames,num_pairs = relative_distances.shape
    num_pt_GPR,_ = GPR_relative_distances.shape

    frame_steps = np.linspace(0,num_frames-1,num_frames,dtype=int)
    GPR_steps = np.linspace(0,num_pt_GPR-1,num_pt_GPR,dtype=int)

    num_rows,num_cols=get_subplot_dims(num_pairs)

    fig,axes = plt.subplots(nrows=num_rows,ncols=num_cols,figsize=(11,7))
    axes = axes.flatten() if num_rows > 1 or num_cols > 1 else [axes]

    for pair_ind in range(num_pairs):

        axes[pair_ind].scatter(frame_steps,relative_distances[:,pair_ind],c='black',s=8,label='relative dist')
        axes[pair_ind].plot(GPR_steps,GPR_relative_distances[:,pair_ind],c='red',linewidth=1,label='GPR smoothed')
        axes[pair_ind].set_title(rel_dist_pair_names[pair_ind])
        axes[pair_ind].set_xlabel('frame number')
        axes[pair_ind].set_ylabel('distance between pillars (pixels)')
        axes[pair_ind].grid('on',ls=':')
    plt.suptitle('Relative Pillar Distances',fontsize=16)

    if is_potential_bg_shift:
        plt.figtext(0.02, 0.99, 
                   r"$\bf{Warning}$: Potential large background shift detected."+"\nPlease verify your experimental images for accuracy.",
                   ha='left', va='top', 
                   fontsize=10, color='red',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

    plt.tight_layout()
    plt.savefig(output_path.joinpath("relative_pillar_distances.png").resolve())
    return


def show_and_save_pillar_positions(
    img: np.ndarray,
    avg_pos_all_x: np.ndarray,
    avg_pos_all_y: np.ndarray,
    output_path: Path,
    title: str = "Pillar Positions"
):
    """Given average pillar positions. Will plot and save the image with pillars."""
    _,num_pillars = avg_pos_all_x.shape
    # dark2_colors = plt.get_cmap("Dark2")
    # colors = [dark2_colors(i) for i in np.linspace(0, 1, num_pillars)]
    colors = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
    plt.figure()
    plt.imshow(img, cmap=plt.cm.gray)
    for pillar_ind in range(num_pillars):
        plt.scatter(avg_pos_all_x[0, pillar_ind], avg_pos_all_y[0, pillar_ind], s=40, label=f'Pillar {pillar_ind}',color=colors[pillar_ind])
    plt.title(title, fontdict={'fontsize': 16})
    plt.legend(loc=(0.5,0.9), fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    save_path = output_path.joinpath("pillar_positions.png").resolve()
    plt.savefig(save_path)
    plt.close()
    return


def run_texture_tracking_pillars(input_path: Path, output_path: Path, threshold_function_idx: int):
    img_list = read_all_tiff(input_path)
    first_img = img_list[0]
    pillar_mask_list = seg.get_pillar_mask_list(first_img, threshold_function_idx)
    avg_pos_all_x, avg_pos_all_y = tt.perform_pillar_tracking(pillar_mask_list, img_list)
    pillar_mask_list, avg_pos_all_x, avg_pos_all_y = pp.rearrange_pillars_indexing(
        pillar_mask_list,avg_pos_all_x,avg_pos_all_y
    )

    is_potential_bg_shift,large_shift_frame_ind = pp.check_potential_large_background_shift(avg_pos_all_x,avg_pos_all_y)
    if is_potential_bg_shift is True:
        folder_path_for_warning = output_path.parent.parent.parent
        sample_name = output_path.parent.name
        basename_exp = output_path.parent.parent.name
        warning_folder_path = create_folder(folder_path_for_warning, "warnings_for_"+basename_exp)
        _ = save_bg_shift_warning(warning_folder_path,sample_name,large_shift_frame_ind)

    relative_distances,rel_dist_pair_names = pp.compute_relative_pillars_dist(avg_pos_all_x,avg_pos_all_y)
    GPR_relative_distances = pp.smooth_relative_pillar_distances_with_GPR(relative_distances)
    show_and_save_relative_pillar_distances(relative_distances,GPR_relative_distances,rel_dist_pair_names,output_path,is_potential_bg_shift)
    show_and_save_pillar_positions(first_img, avg_pos_all_x, avg_pos_all_y, output_path)

    # save data
    path_pos_x = output_path.joinpath("pillar_tracker_x.txt").resolve()
    path_pos_y = output_path.joinpath("pillar_tracker_y.txt").resolve()
    path_relative_distances = output_path.joinpath("relative_pillar_distances.txt").resolve()
    path_relative_distances_GPR = output_path.joinpath("relative_pillar_distances_smoothed_GPR.txt").resolve()
    path_relative_distances_pair_names = output_path.joinpath("relative_pillar_distances_pair_names.txt").resolve()

    np.savetxt(str(path_pos_x), avg_pos_all_x)
    np.savetxt(str(path_pos_y), avg_pos_all_y)
    np.savetxt(str(path_relative_distances), relative_distances)
    np.savetxt(str(path_relative_distances_GPR), GPR_relative_distances)
    np.savetxt(str(path_relative_distances_pair_names), rel_dist_pair_names,fmt="%s")

    return pillar_mask_list, avg_pos_all_x, avg_pos_all_y, path_pos_x, path_pos_y


def save_bg_shift_warning(warning_folder_path: Path, sample_name: str, large_shift_frame_ind: np.ndarray):
    """
    Save a warning text file about potential large background shifts between frames.
    
    Parameters:
        warning_folder_path (Path): Path to the folder where the warning file will be saved
        sample_name (str): Sample name used for the output filename
        large_shift_frame_ind (np.ndarray): Array of frame indices where potential large shifts occur
                                           (1-based indexing, shifts occur between frame_ind and frame_ind+1)
    """
    # Create the output filename
    output_filename = f"{sample_name}_bg_shift_warning.txt"
    output_path = warning_folder_path / output_filename
    
    # Prepare the warning message
    warning_lines = [
        f"Warning: Potential large background shifts detected for sample {sample_name}",
        "------------------------------------------------------------",
        "",
        "The following frame transitions may have large background shifts:",
        "(Note: Frame numbers start from 1)",
        ""
    ]
    
    # Add each shift location to the message
    for frame_ind in large_shift_frame_ind:
        warning_lines.append(f"- Between frame {frame_ind} and frame {frame_ind + 1}")
    
    # Add some additional notes
    warning_lines.extend([
        "",
        "Note: These are potential shifts that may need visual inspection."
    ])
    
    # Combine all lines with newline characters
    warning_text = "\n".join(warning_lines)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(warning_text)
    
    print(f"Warning file saved to: {output_path}")
    return output_path


def show_and_save_tracking(
    img: np.ndarray,
    contour: np.ndarray,
    is_broken: bool,
    is_closed: bool,
    frame: int,
    tracker_x_forward: np.ndarray,
    tracker_y_forward: np.ndarray,
    tracker_x_reverse_forward: np.ndarray,
    tracker_y_reverse_forward: np.ndarray,
    save_path: Path,
    title: str
):
    """Given results of tracking. Will plot a single frame."""
    plt.figure()
    plt.imshow(img, cmap=plt.cm.gray)
    xt = 3.0 * img.shape[1] / 8.0
    yt = 7.0 * img.shape[0] / 8.0
    if is_broken:
        plt.text(xt, yt, "broken", color="r", backgroundcolor="w", fontsize=20)
    else:
        if is_closed:
            if contour is not None:
                plt.plot(contour[:, 1], contour[:, 0], 'k', linewidth=2.0, antialiased=True)
        else:
            if contour is not None:
                plt.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2.0, antialiased=True)
        # plot tracked points
        plt.plot(tracker_x_forward[:, 0:frame].T, tracker_y_forward[:, 0:frame].T, "y-")
        plt.plot(tracker_x_forward[:, frame], tracker_y_forward[:, frame], "y.")
        plt.plot(tracker_x_reverse_forward[:, 0:frame].T, tracker_y_reverse_forward[:, 0:frame].T, "c-")
        plt.plot(tracker_x_reverse_forward[:, frame], tracker_y_reverse_forward[:, frame], "c.")
    plt.title(title,fontdict={'fontsize': 16})
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return


def save_all_img_tracking(
    output_path: Path,
    fname: str,
    img_list: List,
    contour_list: List,
    is_broken_list: List,
    is_closed_list: List,
    tracker_x_forward: np.ndarray,
    tracker_y_forward: np.ndarray,
    tracker_x_reverse_forward: np.ndarray,
    tracker_y_reverse_forward: np.ndarray
):
    """"Given the results of tracking all frames. Will plot all frames."""
    file_name_list = []
    for kk in range(0, len(img_list)):
        img = img_list[kk]
        contour = contour_list[kk]
        is_broken = is_broken_list[kk]
        is_closed = is_closed_list[kk]
        title = "frame %05d" % (kk)
        frame = kk
        save_path = output_path.joinpath(fname + "_%05d.png" % (kk)).resolve()
        show_and_save_tracking(img, contour, is_broken, is_closed, frame, tracker_x_forward, tracker_y_forward, tracker_x_reverse_forward, tracker_y_reverse_forward, save_path, title)
        file_name_list.append(save_path)
    return file_name_list


def run_texture_tracking_visualize(
    output_path: Path,
    img_list: List,
    contour_list: List,
    is_broken_list: List,
    is_closed_list: List,
    tracker_x_forward: np.ndarray,
    tracker_y_forward: np.ndarray,
    tracker_x_reverse_forward: np.ndarray,
    tracker_y_reverse_forward: np.ndarray
) -> tuple:
    """Visualize tracking results -- still images and gif."""
    fname = "tracking"
    path_list = save_all_img_tracking(output_path, fname, img_list, contour_list, is_broken_list, is_closed_list, tracker_x_forward, tracker_y_forward, tracker_x_reverse_forward, tracker_y_reverse_forward)
    gif_path = create_gif(output_path, fname, path_list)
    return (path_list, gif_path)


def load_contour_coords(folder_path: Path):
    num_files = len(glob.glob(str(folder_path) + "/ph1_images/*.TIF")) + len(glob.glob(str(folder_path) + "/ph1_images/*.tiff"))
    contour_coords_list = []
    for kk in range(0, num_files):
        fname = str(folder_path) + "/segment_ph1/contour_coords_%05d.npy" % (kk)
        if len(glob.glob(fname)) > 0:
            contour_coords_list.append(np.load(fname))
        else:
            contour_coords_list.append(None)
    return contour_coords_list


def combine_images(
    folder_path:Path, 
    output_path:Path,
    image_type:str,
    max_combined_width:int=2000,
    individual_max_size:int=300):
    """
    Combine images from a folder into a grid, with frame numbers labeled.
    
    Args:
        folder_path (str): Path to folder containing images
        output_path (str): Path to save the combined image (default: "combined.png")
        max_combined_width (int): Maximum width of the combined image (default: 2000)
        individual_max_size (int): Maximum size of individual images in the combined output (default: 300)
    """
    
    # Get all PNG files in the folder and sort by frame number
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png') and "_all" not in f]
    image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Obtain the sample name from the folder path
    folder_path_str = str(folder_path)
    normalized_path = os.path.normpath(folder_path_str)
    segments = [s for s in normalized_path.split(os.path.sep) if s]
    
    if "visualizations" not in segments:
        print("The folder path does not contain the 'visualizations' folder. Skip combining ph1_contour_*.png images.")
        return
    else:
        vis_index = segments.index("visualizations")
        sample_name = segments[vis_index - 2]

    output_filename = image_type +"_all_"+sample_name+".png"
    
    # Calculate grid dimensions (as close to square as possible)
    num_images = len(image_files)
    grid_cols = np.ceil(np.sqrt(num_images))
    grid_rows = np.ceil(num_images / grid_cols)
    
    # Load first image to get dimensions
    first_image = Image.open(os.path.join(folder_path, image_files[0]))
    original_width, original_height = first_image.size
    
    # Calculate scaling factor to fit within individual_max_size while maintaining aspect ratio
    scale_factor = min(individual_max_size / original_width, individual_max_size / original_height)
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # Calculate padding (10% of individual image width)
    padding = int(new_width * 0.1)
    
    # Calculate combined image dimensions
    combined_width = int(grid_cols * (new_width + padding) + padding)
    combined_height = int(grid_rows * (new_height + padding) + padding)
    
    # Create blank combined image
    combined_image = Image.new('RGB', (combined_width, combined_height), color='white')
    
    # Process each image
    for i, image_file in enumerate(image_files):
        # Calculate grid position
        row = i // grid_cols
        col = i % grid_cols
        
        # Open and resize image
        img = Image.open(os.path.join(folder_path, image_file))
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Calculate position in combined image
        x = int(col * (new_width + padding) + padding)
        y = int(row * (new_height + padding) + padding)
        
        # Paste image
        combined_image.paste(img, (x, y))
    
    # Save combined image
    combined_image.save(output_path.joinpath(output_filename).resolve())
    return


def check_before_injury_folder(folder_path: Path) -> bool:
    """
    Check if the folder path contains a before injury ("_bi") folder.
    
    Args:
        folder_path (Path): Path to the folder to check.
        
    Returns:
        bool: True if the path contains "_bi" , False otherwise.
    """
    # Check if the folder path contains '_bi' indicating before injury
    s = str(folder_path)
    if re.search(r'_bi(?![a-zA-Z])', s):
        return True
    else:
        return False


def run_all(folder_path: Path) -> List:
    """Given a folder path. Will read input, run code, generate all outputs."""
    time_all = []
    action_all = []
    time_all.append(time.time())
    action_all.append("start")
    input_dict, input_path_dict, output_path_dict = input_info_to_dicts(folder_path)
    time_all.append(time.time())
    action_all.append("loaded input")
    zoom_fcn = com.select_zoom_function(input_dict)
    is_in_bi_folder = check_before_injury_folder(folder_path)

    if input_dict["segment_brightfield"] is True:
        input_path = input_path_dict["brightfield_images_path"]
        output_path = output_path_dict["segment_brightfield_path"]
        thresh_fcn = seg.select_threshold_function(input_dict, True, False, False)
        # throw errors here if input_path == None? (future) and/or output dir isn't created
        _, _, _, _, _, _, _, _, _, img_list_bf, contour_list_bf, tissue_param_list_bf, is_broken_list_bf, is_closed_list_bf = run_segment(input_path, output_path, thresh_fcn, zoom_fcn)
        time_all.append(time.time())
        action_all.append("segmented brightfield")
    if input_dict["segment_fluorescent"] is True:
        input_path = input_path_dict["fluorescent_images_path"]
        output_path = output_path_dict["segment_fluorescent_path"]
        thresh_fcn = seg.select_threshold_function(input_dict, False, True, False)
        # throw errors here if input_path == None? (future) and/or output dir isn't created
        _, _, _, _, _, _, _, _, _, img_list_fl, contour_list_fl, tissue_param_list_fl, is_broken_list_fl, is_closed_list_fl = run_segment(input_path, output_path, thresh_fcn, zoom_fcn)
        time_all.append(time.time())
        action_all.append("segmented fluorescent")
    if input_dict["track_pillars_ph1"] is True:
        output_path = output_path_dict["track_pillars_ph1_path"]
        input_path = input_path_dict["ph1_images_path"]
        img_list_ph1 = read_all_tiff(input_path)
        thresh_fcn = seg.select_threshold_function(input_dict, False, False, True)
        pillar_masks_list,avg_pos_all_x,avg_pos_all_y,_, _ = run_texture_tracking_pillars(input_path, output_path, thresh_fcn)
        time_all.append(time.time())
        action_all.append("run pillar texture tracking")
    else:
        avg_pos_all_x = None
        avg_pos_all_y = None
        pillar_masks_list = None
    if input_dict["segment_ph1"] is True:
        input_path = input_path_dict["ph1_images_path"]
        output_path = output_path_dict["segment_ph1_path"]
        thresh_fcn = seg.select_threshold_function(input_dict, False, False, True)
        # throw errors here if input_path == None? (future) and/or output dir isn't created
        _, _, _, _, _, _, _, _, _, img_list_ph1, contour_list_ph1, tissue_param_list_ph1, is_broken_list_ph1, is_closed_list_ph1 = run_segment(input_path, output_path, thresh_fcn, zoom_fcn, avg_pos_all_x,avg_pos_all_y,is_in_bi_folder)
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
        _ = run_seg_visualize(output_path,img_list_ph1,contour_list_ph1,tissue_param_list_ph1,is_broken_list_ph1,is_closed_list_ph1,fname,avg_pos_all_x,avg_pos_all_y,pillar_masks=pillar_masks_list,is_in_bi_folder=is_in_bi_folder)
        combine_images(folder_path=output_path,output_path=output_path,image_type=fname)
        # throw errors here if necessary segmentation data doesn't exist
        time_all.append(time.time())
        action_all.append("visualized ph1")
    if input_dict["bf_seg_with_fl_seg_visualize"] is True:
        output_path = output_path_dict["bf_seg_with_fl_seg_visualize_path"]
        _ = run_bf_seg_vs_fl_seg_visualize(output_path, img_list_bf, contour_list_bf, contour_list_fl)
        # throw errors here if necessary segmentation data doesn't exist
        time_all.append(time.time())
        action_all.append("visualized brightfield and fluorescent")
    if input_dict["track_ph1"] is True:
        input_path = input_path_dict["ph1_images_path"]
        output_path = output_path_dict["track_ph1_path"]
        thresh_fcn = seg.select_threshold_function(input_dict, False, False, True)
        tracker_x_forward, tracker_y_forward, tracker_x_reverse_forward, tracker_y_reverse_forward, _, _, _, _, _, _, _, _ = run_texture_tracking(input_path, output_path, thresh_fcn)
        time_all.append(time.time())
        action_all.append("run texture tracking")
    if input_dict["track_ph1_visualize"] is True:
        output_path = output_path_dict["track_ph1_vis_path"]
        input_path = input_path_dict["ph1_images_path"]
        img_list_ph1 = read_all_tiff(input_path)
        contour_list_ph1 = load_contour_coords(folder_path)
        is_broken_list_ph1 = list(np.loadtxt(str(folder_path) + "/segment_ph1/is_broken_vs_frame.txt"))
        is_closed_list_ph1 = list(np.loadtxt(str(folder_path) + "/segment_ph1/is_closed_vs_frame.txt"))
        _ = run_texture_tracking_visualize(output_path, img_list_ph1, contour_list_ph1, is_broken_list_ph1, is_closed_list_ph1, tracker_x_forward, tracker_y_forward, tracker_x_reverse_forward, tracker_y_reverse_forward)
        time_all.append(time.time())
        action_all.append("visualized texture tracking")
    return time_all, action_all
