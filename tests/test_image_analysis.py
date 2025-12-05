import copy
import glob
import numpy as np
from pathlib import Path
import pytest
from skimage import io,draw
from woundcompute import compute_values as com
from woundcompute import image_analysis as ia
from woundcompute import segmentation as seg
from woundcompute import texture_tracking as tt
from woundcompute import post_process as pp


def files_path():
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    return data_path


def example_path(example_name):
    data_path = files_path()
    example_path = data_path.joinpath(example_name).resolve()
    return example_path


def glob_brightfield(example_name):
    folder_path = example_path(example_name)
    bf_path = folder_path.joinpath("brightfield_images").resolve()
    name_list = glob.glob(str(bf_path) + '/*.TIF')
    name_list.sort()
    name_list_path = []
    for name in name_list:
        name_list_path.append(Path(name))
    return name_list


def glob_fluorescent(example_name):
    folder_path = example_path(example_name)
    fl_path = folder_path.joinpath("fluorescent_images").resolve()
    name_list = glob.glob(str(fl_path) + '/*.TIF')
    name_list.sort()
    name_list_path = []
    for name in name_list:
        name_list_path.append(Path(name))
    return name_list


def glob_ph1(example_name):
    folder_path = example_path(example_name)
    fl_path = folder_path.joinpath("ph1_images").resolve()
    name_list = glob.glob(str(fl_path) + '/*.TIF')
    name_list.sort()
    name_list_path = []
    for name in name_list:
        name_list_path.append(Path(name))
    return name_list


def glob_yaml(example_name):
    folder_path = example_path(example_name)
    # yaml_name_list = glob.glob(str(folder_path) + '/*.yaml') + glob.glob(str(folder_path) + '/*.yml')
    # return Path(yaml_name_list[0])
    yaml_path = folder_path.joinpath(example_name + ".yaml").resolve()
    return yaml_path


def yaml_test(test_name):
    folder_path = example_path("test_io")
    output_path = folder_path.joinpath(test_name).resolve()
    return output_path


def output_file(example_name, file_name):
    folder_path = example_path(example_name)
    output_path = folder_path.joinpath(file_name).resolve()
    return output_path


def test_hello_world():
    res = ia.hello_wound_compute()
    assert res == "Hello World!"


def test_read_tiff():
    file_path = glob_brightfield("test_single")[0]
    known = io.imread(file_path)
    found = ia.read_tiff(file_path)
    assert np.allclose(known, found)


def test_show_and_save_image():
    file_path = glob_brightfield("test_single")[0]
    file = ia.read_tiff(file_path)
    save_path = output_file("test_single", "test_brightfield_save_no_title.png")
    ia.show_and_save_image(file, save_path)
    assert save_path.is_file()
    save_path_title = output_file("test_single", "test_brightfield_save_title.png")
    title = 'test brightfield title'
    ia.show_and_save_image(file, save_path_title, title)
    assert save_path_title.is_file()


def test_show_and_save_image_mask_bf():
    file_path = glob_brightfield("test_single")[0]
    file = ia.read_tiff(file_path)
    file_thresh = seg.threshold_array(file, 1)
    tissue_mask, wound_mask, _ = seg.isolate_masks(file_thresh, 1)
    save_path = output_file("test_single", "test_brightfield_tissue_mask.png")
    ia.show_and_save_image(tissue_mask, save_path)
    assert save_path.is_file()
    save_path = output_file("test_single", "test_brightfield_wound_mask.png")
    ia.show_and_save_image(wound_mask, save_path)
    assert save_path.is_file()


def test_show_and_save_image_mask_fl():
    file_path = glob_fluorescent("test_single")[0]
    file = ia.read_tiff(file_path)
    file_thresh = seg.threshold_array(file, 2)
    tissue_mask, wound_mask, _ = seg.isolate_masks(file_thresh, 2)
    save_path = output_file("test_single", "test_gfp_tissue_mask.png")
    ia.show_and_save_image(tissue_mask, save_path)
    assert save_path.is_file()
    save_path = output_file("test_single", "test_gfp_wound_mask.png")
    ia.show_and_save_image(wound_mask, save_path)
    assert save_path.is_file()


def test_show_and_save_image_mask_ph1():
    file_path = glob_ph1("test_single")[0]
    file = ia.read_tiff(file_path)
    file_thresh = seg.threshold_array(file, 3)
    save_path = output_file("test_single", "test_ph1_orig_mask.png")
    ia.show_and_save_image(file_thresh, save_path)
    tissue_mask, wound_mask, _ = seg.isolate_masks(file_thresh, 3)
    save_path = output_file("test_single", "test_ph1_tissue_mask.png")
    ia.show_and_save_image(tissue_mask, save_path)
    assert save_path.is_file()
    save_path = output_file("test_single", "test_ph1_wound_mask.png")
    ia.show_and_save_image(wound_mask, save_path)
    assert save_path.is_file()


def test_show_and_save_contour():
    file_path = glob_brightfield("test_single")[0]
    file = ia.read_tiff(file_path)
    file_thresh = seg.threshold_array(file, 1)
    wound_mask = seg.isolate_masks(file_thresh, 1)[1]
    contour = seg.mask_to_contour(wound_mask)
    save_path = output_file("test_single", "test_brightfield_wound_contour.png")
    is_broken = False
    is_closed = False
    ia.show_and_save_contour(file, contour, is_broken, is_closed, save_path)
    save_path = output_file("test_single", "test_brightfield_wound_contour_broken_label.png")
    is_broken = True
    is_closed = False
    ia.show_and_save_contour(file, contour, is_broken, is_closed, save_path)
    save_path = output_file("test_single", "test_brightfield_wound_contour_closed_label.png")
    is_broken = False
    is_closed = True
    ia.show_and_save_contour(file, contour, is_broken, is_closed, save_path)
    assert save_path.is_file()
    save_path = output_file("test_single", "test_brightfield_wound_contour_no_contour.png")
    is_broken = False
    is_closed = False
    contour = None
    ia.show_and_save_contour(file, contour, is_broken, is_closed, save_path)
    assert save_path.is_file()


def test_show_and_save_tissue_wound_pillar_contours():
    file_path = glob_brightfield("test_single")[0]
    file = ia.read_tiff(file_path)
    selection_idx = 1
    zoom_fcn_idx = 1
    file_thresh = seg.threshold_array(file, selection_idx)
    tissue_contour = None
    wound_mask = seg.isolate_masks(file_thresh, selection_idx)[1]
    wound_contour = seg.mask_to_contour(wound_mask)
    save_path = output_file("test_single", "test_brightfield_wound_contour.png")
    is_broken = False
    is_closed = False
    thresholded_list = [file_thresh]
    tissue_mask_list, wound_mask_list, _ = seg.mask_all(thresholded_list, selection_idx)
    tissue_parameters = com.tissue_parameters_all([tissue_mask_list[0]], [wound_mask_list[0]], zoom_fcn_idx)[0]
    points = [[tissue_parameters[1], tissue_parameters[3]], [tissue_parameters[2], tissue_parameters[4]]]
    ia.show_and_save_tissue_wound_pillar_contours(file, tissue_contour, wound_contour, is_broken, is_closed, points, save_path)
    assert save_path.is_file()
    save_path = output_file("test_single", "test_brightfield_wound_contour_broken_label.png")
    is_broken = True
    is_closed = False
    ia.show_and_save_tissue_wound_pillar_contours(file, tissue_contour, wound_contour, is_broken, is_closed, points, save_path)
    assert save_path.is_file()
    save_path = output_file("test_single", "test_brightfield_wound_contour_closed_label.png")
    is_broken = False
    is_closed = True
    ia.show_and_save_tissue_wound_pillar_contours(file, tissue_contour, wound_contour, is_broken, is_closed, points, save_path)
    assert save_path.is_file()
    save_path = output_file("test_single", "test_brightfield_wound_contour_no_labels.png")
    is_broken = False
    is_closed = False
    wound_contour = None
    points = None
    ia.show_and_save_tissue_wound_pillar_contours(file, tissue_contour, wound_contour, is_broken, is_closed, points, save_path)
    assert save_path.is_file()


def test_show_and_save_tissue_wound_pillar_contours_ph1():
    file_path = glob_ph1("test_single")[0]
    file = ia.read_tiff(file_path)
    selection_idx = 4
    zoom_fcn_idx = 2
    file_thresh = seg.threshold_array(file, selection_idx)
    wound_mask = seg.isolate_masks(file_thresh, selection_idx)[1]
    wound_contour = seg.mask_to_contour(wound_mask)
    save_path = output_file("test_single", "test_ph1_contour_and_width.png")
    is_broken = False
    is_closed = False
    thresholded_list = [file_thresh]
    tissue_mask_list, wound_mask_list, _ = seg.mask_all(thresholded_list, selection_idx)
    tissue_contour = seg.mask_to_contour(tissue_mask_list[0])
    tissue_parameters = com.tissue_parameters_all([tissue_mask_list[0]], [wound_mask_list[0]], zoom_fcn_idx)[0]
    points = [[tissue_parameters[1], tissue_parameters[3]], [tissue_parameters[2], tissue_parameters[4]]]
    ia.show_and_save_tissue_wound_pillar_contours(file, tissue_contour, wound_contour, is_broken, is_closed, points, save_path)
    assert save_path.is_file()


def test_show_and_save_tissue_wound_pillar_contours_ph1_high_res():
    file_path = glob_ph1("test_ph1_movie_mini_large_bg_shift")[0]
    file = ia.read_tiff(file_path)
    selection_idx = 4
    zoom_fcn_idx = 2
    file_thresh = seg.threshold_array(file, selection_idx)
    wound_mask = seg.isolate_masks(file_thresh, selection_idx)[1]
    wound_contour = seg.mask_to_contour(wound_mask)
    save_path = output_file("test_single", "test_high_res_im.png")
    is_broken = False
    is_closed = False
    thresholded_list = [file_thresh]
    tissue_mask_list, wound_mask_list, _ = seg.mask_all(thresholded_list, selection_idx)
    tissue_contour = seg.mask_to_contour(tissue_mask_list[0])
    tissue_parameters = com.tissue_parameters_all([tissue_mask_list[0]], [wound_mask_list[0]], zoom_fcn_idx)[0]
    points = [[tissue_parameters[1], tissue_parameters[3]], [tissue_parameters[2], tissue_parameters[4]]]
    ia.show_and_save_tissue_wound_pillar_contours(file, tissue_contour, wound_contour, is_broken, is_closed, points, save_path)
    assert save_path.is_file()


def test_show_and_save_bi_tissue():
    file_path = glob_ph1("test_before_injury")[0]
    file = ia.read_tiff(file_path)
    is_broken=False
    save_path = output_file("test_before_injury", "test_ph1_tissue_mask_bi.png")
    ia.show_and_save_bi_tissue(file,is_broken,save_path,0,"test_before_injury")


def test_show_and_save_bs_tissue():
    file_path = glob_ph1("test_before_seeding")[0]
    file = ia.read_tiff(file_path)
    save_path = output_file("test_before_seeding","test_ph1_bs.png")
    ia.show_and_save_bs_tissue(file,save_path,"test_before_seeding")


def test_show_and_save_double_contour():
    file_path = glob_brightfield("test_single")[0]
    file = ia.read_tiff(file_path)
    file_thresh = seg.threshold_array(file, 1)
    wound_mask = seg.isolate_masks(file_thresh, 1)[1]
    contour_bf = seg.mask_to_contour(wound_mask)
    file_path = glob_fluorescent("test_single")[0]
    file = ia.read_tiff(file_path)
    file_thresh = seg.threshold_array(file, 2)
    wound_mask = seg.isolate_masks(file_thresh, 2)[1]
    contour_fl = seg.mask_to_contour(wound_mask)
    save_path = output_file("test_single", "test_brightfield_and_fluorescent_wound_contour.png")
    ia.show_and_save_double_contour(file, contour_bf, contour_fl, save_path)
    assert save_path.is_file()


def test_save_numpy():
    data_path = files_path()
    file_path = glob_brightfield("test_single")[0]
    file = ia.read_tiff(file_path)
    save_path = data_path.joinpath("test_brightfield_save_no_title.npy")
    ia.save_numpy(file, save_path)
    assert save_path.is_file()
    file_thresh = seg.threshold_array(file, 1)
    tissue_mask, wound_mask, _ = seg.isolate_masks(file_thresh, 1)
    save_path = data_path.joinpath("test_brightfield_tissue_mask.npy")
    ia.save_numpy(tissue_mask, save_path)
    assert save_path.is_file()
    contour = seg.mask_to_contour(wound_mask)
    save_path = data_path.joinpath("test_brightfield_tissue_contour.npy")
    ia.save_numpy(contour, save_path)
    assert save_path.is_file()


def test_yml_to_dict():
    input_file_path = glob_yaml("test_single")
    db = ia._yml_to_dict(yml_path_file=input_file_path)
    assert db["version"] == 1.0
    assert db["segment_brightfield"] is True
    assert db["seg_bf_version"] == 1
    assert db["seg_bf_visualize"] is False
    assert db["segment_fluorescent"] is True
    assert db["seg_fl_version"] == 1
    assert db["seg_fl_visualize"] is False
    assert db["track_brightfield"] is False
    assert db["track_bf_version"] == 1
    assert db["track_bf_visualize"] is False
    assert db["bf_seg_with_fl_seg_visualize"] is False
    assert db["bf_track_with_fl_seg_visualize"] is False
    assert db["track_pillars_ph1"] is False
    assert db["segment_dic"] is False
    assert db["seg_dic_version"] == 1
    assert db["seg_dic_visualize"] is False
    assert db["track_dic_visualize"] is False
    assert db["track_pillars_dic"] is False
    assert db["frame_inds_to_skip"] == []


def test_create_folder():
    folder_path = example_path("test_io")
    new_folder_name = "test_create_folder"
    new_folder = ia.create_folder(folder_path, new_folder_name)
    assert new_folder.is_dir()


def test_input_info_to_input_dict():
    folder_path = example_path("test_single")
    db = ia.input_info_to_input_dict(folder_path)
    assert db["version"] == 1.0
    assert db["segment_brightfield"] is True
    assert db["seg_bf_version"] == 1
    assert db["seg_bf_visualize"] is False
    assert db["segment_fluorescent"] is True
    assert db["seg_fl_version"] == 1
    assert db["seg_fl_visualize"] is False
    assert db["track_brightfield"] is False
    assert db["track_bf_version"] == 1
    assert db["track_bf_visualize"] is False
    assert db["bf_seg_with_fl_seg_visualize"] is False
    assert db["bf_track_with_fl_seg_visualize"] is False
    assert db["track_pillars_ph1"] is False


def test_input_info_to_input_paths():
    folder_path = example_path("test_single")
    path_dict = ia.input_info_to_input_paths(folder_path)
    assert path_dict["brightfield_images_path"].is_dir()
    assert path_dict["fluorescent_images_path"].is_dir()
    folder_path = example_path("test_io")
    path_dict = ia.input_info_to_input_paths(folder_path)
    assert path_dict["brightfield_images_path"] is None
    assert path_dict["fluorescent_images_path"] is None


def test_image_folder_to_path_list():
    folder_path = example_path("test_mini_movie")
    path_dict = ia.input_info_to_input_paths(folder_path)
    path_list = ia.image_folder_to_path_list(path_dict["brightfield_images_path"])
    assert len(path_list) == 5
    assert str(path_list[-1])[-8:] == "0020.TIF"


def test_when_io_fails():
    # If the user tries to run with a file that does not exist
    # then check that a FileNotFoundError is raised
    with pytest.raises(FileNotFoundError) as error:
        input_file = yaml_test("this_file_does_not_exist.yml")
        ia._yml_to_dict(yml_path_file=input_file)
    assert error.typename == "FileNotFoundError"

    # If the user tries to run with a file type that is not a .yml or .yaml,
    # then check that a TypeError is raised.
    with pytest.raises(TypeError) as error:
        input_file = yaml_test("wrong_file_type.txt")
        ia._yml_to_dict(yml_path_file=input_file)
    assert error.typename == "TypeError"

    # If the user tried to run the input yml version that is not the version
    # curently implemented, then check that a ValueError is raised.
    with pytest.raises(ValueError) as error:
        input_file = yaml_test("wrong_version.yaml")
        ia._yml_to_dict(yml_path_file=input_file)
    assert error.typename == "ValueError"

    # If the user tried to run the input yml that
    # does not have the correct keys, then test that a KeyError is raised.
    with pytest.raises(KeyError) as error:
        input_file = yaml_test("bad_keys.yaml")
        ia._yml_to_dict(yml_path_file=input_file)
    assert error.typename == "KeyError"

    # If the yaml cannot be loaded, then test that an OSError is raised.
    with pytest.raises(OSError) as error:
        input_file = yaml_test("bad_load.yaml")
        ia._yml_to_dict(yml_path_file=input_file)
    assert error.typename == "OSError"


def test_create_folder_guaranteed_conditions():
    folder_path = example_path("test_io")
    new_folder_name = "test_create_folder_%i" % (np.random.random() * 1000000)
    new_folder = ia.create_folder(folder_path, new_folder_name)
    assert new_folder.is_dir()
    new_folder = ia.create_folder(folder_path, new_folder_name)
    assert new_folder.is_dir()


def test_input_info_to_output_folders_create_all():
    io_path = example_path("test_io")
    folder_path = io_path.joinpath("all_true").resolve()
    yaml_path = folder_path.joinpath("all_true.yaml").resolve()
    input_dict = ia._yml_to_dict(yml_path_file=yaml_path)
    path_dict = ia.input_info_to_output_paths(folder_path, input_dict)
    for key in path_dict:
        assert path_dict[key].is_dir()


def test_image_folder_to_path_list_tif():
    io_path = example_path("test_io")
    folder_path = io_path.joinpath("tif_lowercase").resolve()
    path_dict = ia.input_info_to_input_paths(folder_path)
    path_list = ia.image_folder_to_path_list(path_dict["brightfield_images_path"])
    assert len(path_list) == 5
    assert str(path_list[-1])[-8:] == "0020.tif"


def test_input_info_to_output_folders_create_none():
    io_path = example_path("test_io")
    folder_path = io_path.joinpath("all_false").resolve()
    yaml_path = folder_path.joinpath("all_false.yaml").resolve()
    input_dict = ia._yml_to_dict(yml_path_file=yaml_path)
    path_dict = ia.input_info_to_output_paths(folder_path, input_dict)
    for key in path_dict:
        assert path_dict[key] is None


def test_input_info_to_dicts():
    folder_path = example_path("test_single")
    db, input_path_dict, output_path_dict = ia.input_info_to_dicts(folder_path)
    assert db["version"] == 1.0
    assert db["segment_brightfield"] is True
    assert db["seg_bf_version"] == 1
    assert db["seg_bf_visualize"] is False
    assert db["segment_fluorescent"] is True
    assert db["seg_fl_version"] == 1
    assert db["seg_fl_visualize"] is False
    assert db["track_brightfield"] is False
    assert db["track_bf_version"] == 1
    assert db["track_bf_visualize"] is False
    assert db["bf_seg_with_fl_seg_visualize"] is False
    assert db["bf_track_with_fl_seg_visualize"] is False
    assert input_path_dict["brightfield_images_path"].is_dir()
    assert input_path_dict["fluorescent_images_path"].is_dir()
    assert output_path_dict["segment_brightfield_path"].is_dir()
    assert output_path_dict["segment_brightfield_vis_path"] is None
    assert output_path_dict["segment_fluorescent_path"].is_dir()
    assert output_path_dict["segment_fluorescent_vis_path"] is None
    assert output_path_dict["track_brightfield_path"] is None
    assert output_path_dict["track_brightfield_vis_path"] is None
    assert output_path_dict["bf_seg_with_fl_seg_visualize_path"] is None
    assert output_path_dict["bf_track_with_fl_seg_visualize_path"] is None


def test_read_all_tiff():
    folder_path = example_path("test_mini_movie")
    path_dict = ia.input_info_to_input_paths(folder_path)
    folder_path = path_dict["brightfield_images_path"]
    tiff_list = ia.read_all_tiff(folder_path)
    assert len(tiff_list) == 5
    assert tiff_list[0].shape == (512, 512)


def test_save_all_numpy():
    folder_path = example_path("test_io")
    file_name = "test_save_numpy"
    array_list = []
    for kk in range(0, 3):
        array_list.append((np.random.random((5, 5))))
    array_list.append(None)
    file_name_list = ia.save_all_numpy(folder_path, file_name, array_list)
    for kk in range(0, 3):
        file_name = file_name_list[kk]
        assert file_name.is_file()
    assert file_name_list[3].is_file() is False


def test_save_list():
    folder_path = example_path("test_io")
    file_name = "test_save_list"
    value_list = [1, 2, 3, 4, 5]
    saved_name = ia.save_list(folder_path, file_name, value_list)
    assert saved_name.is_file()
    folder_path = example_path("test_io")
    file_name = "test_save_list_none"
    value_list = [None, 2, 3, 4, 5]
    saved_name = ia.save_list(folder_path, file_name, value_list)
    assert saved_name.is_file()


def test_subtract_moving_pillar_from_tissue():
    tissue_masks = np.zeros((3,100,100),dtype=np.uint8)
    tissue_masks[:,20:80,20:80]=1
    pillar_mask = np.zeros((100,100),dtype=np.uint8)
    pillar_mask[30:41,30:41]=1
    pillar_centers = np.array([[35,35],[35,35],[35,35]])

    known_masks = copy.deepcopy(tissue_masks)
    known_masks[:,29:41,29:41]=0
    found_masks = ia.subtract_moving_pillar_from_tissue(tissue_masks,pillar_mask,pillar_centers)
    assert np.allclose(known_masks,found_masks,1)


def test_subtract_moving_pillars_from_tissue_masks():

    file_path = glob_ph1("test_single")[0]
    file = ia.read_tiff(file_path)
    img_list = [file,file]
    selection_idx = 4
    file_thresh = seg.threshold_array(file, selection_idx)
    thresholded_list = [file_thresh,file_thresh]

    tissue_mask_list, _, _ = seg.mask_all(thresholded_list, selection_idx)
    pillar_mask_list,res_func = seg.get_pillar_mask_list(file,4,2)
    avg_pos_x,avg_pos_y=tt.perform_pillar_tracking(pillar_mask_list,img_list,2,res_func)
    tissue_mask_list_new = ia.subtract_moving_pillars_from_tissue_masks(tissue_mask_list,pillar_mask_list,avg_pos_x,avg_pos_y)

    assert len(tissue_mask_list_new) == len(tissue_mask_list)


def test_thresh_all():
    folder_path = example_path("test_mini_movie")
    path_dict = ia.input_info_to_input_paths(folder_path)
    folder_path = path_dict["brightfield_images_path"]
    tiff_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = 1
    thresholded_list = seg.threshold_all(tiff_list, threshold_function_idx)
    assert len(thresholded_list) == 5
    for img in thresholded_list:
        assert np.max(img) == 1
        assert np.min(img) == 0


def test_mask_all():
    folder_path = example_path("test_mini_movie")
    path_dict = ia.input_info_to_input_paths(folder_path)
    folder_path = path_dict["brightfield_images_path"]
    tiff_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = 1
    thresholded_list = seg.threshold_all(tiff_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, wound_region_list = seg.mask_all(thresholded_list, 1)
    assert len(tissue_mask_list) == 5
    assert len(wound_mask_list) == 5
    assert len(wound_region_list) == 5
    for img in tissue_mask_list:
        assert np.max(img) == 1
        assert np.min(img) == 0
    for img in wound_mask_list:
        assert np.max(img) == 1
        assert np.min(img) == 0


def test_run_segment_bf():
    for name in ["test_single", "test_mini_movie"]:
        for kind in ["brightfield"]:
            print(name, kind)
            folder_path = example_path(name)
            path_dict = ia.input_info_to_input_paths(folder_path)
            input_path = path_dict[kind + "_images_path"]
            input_dict = ia.input_info_to_input_dict(folder_path)
            path_dict = ia.input_info_to_output_paths(folder_path, input_dict)
            output_path = path_dict["segment_" + kind + "_path"]
            threshold_function_idx = 1
            zoom_fcn_idx = 1
            wound_name_list, tissue_name_list, wound_contour_name_list, area_path, ax_maj_path, ax_min_path, tissue_path, is_broken_path, is_closed_path, _, _, _, _, _, _ = ia.run_segment(input_path, output_path, threshold_function_idx, zoom_fcn_idx)
            for wn in wound_name_list:
                assert wn.is_file()
            for tn in tissue_name_list:
                assert tn.is_file()
            for cn in wound_contour_name_list:
                assert cn.is_file()
            assert tissue_path.is_file()
            assert area_path.is_file()
            assert ax_maj_path.is_file()
            assert ax_min_path.is_file()
            assert is_broken_path.is_file()
            assert is_closed_path.is_file()


def test_run_segment_fl():
    for name in ["test_mini_movie"]:
        for kind in ["fluorescent"]:
            print(name, kind)
            folder_path = example_path(name)
            path_dict = ia.input_info_to_input_paths(folder_path)
            input_path = path_dict[kind + "_images_path"]
            input_dict = ia.input_info_to_input_dict(folder_path)
            path_dict = ia.input_info_to_output_paths(folder_path, input_dict)
            output_path = path_dict["segment_" + kind + "_path"]
            threshold_function_idx = 2
            zoom_fcn_idx = 1
            wound_name_list, tissue_name_list, contour_name_list, area_path, ax_maj_path, ax_min_path, tissue_path, is_broken_path, is_closed_path, _, _, _, _, _, _ = ia.run_segment(input_path, output_path, threshold_function_idx, zoom_fcn_idx)
            for wn in wound_name_list:
                assert wn.is_file()
            for tn in tissue_name_list:
                assert tn.is_file()
            for cn in contour_name_list:
                assert cn.is_file()
            assert area_path.is_file()
            assert ax_maj_path.is_file()
            assert ax_min_path.is_file()
            assert tissue_path.is_file()
            assert is_broken_path.is_file()
            assert is_closed_path.is_file()


def test_run_segment_ph1():
    name = "test_ph1_mini_movie"
    kind = "ph1"
    folder_path = example_path(name)
    path_dict = ia.input_info_to_input_paths(folder_path)
    input_path = path_dict[kind + "_images_path"]
    input_dict = ia.input_info_to_input_dict(folder_path)
    path_dict = ia.input_info_to_output_paths(folder_path, input_dict)
    output_path = path_dict["segment_" + kind + "_path"]
    threshold_function_idx = 3
    zoom_function_idx = 1
    wound_name_list, tissue_name_list, contour_name_list, area_path, ax_maj_path, ax_min_path, tissue_path, is_broken_path, is_closed_path, _, _, _, _, _, _ = ia.run_segment(input_path, output_path, threshold_function_idx, zoom_function_idx)
    for wn in wound_name_list:
        assert wn.is_file()
    for tn in tissue_name_list:
        assert tn.is_file()
    for cn in contour_name_list:
        assert cn.is_file()
    assert area_path.is_file()
    assert ax_maj_path.is_file()
    assert ax_min_path.is_file()
    assert tissue_path.is_file()
    assert is_broken_path.is_file()
    assert is_closed_path.is_file()


def test_run_segment_with_pillar_info():
    name = "test_phi_movie_mini_Anish_tracking"
    kind = "ph1"
    folder_path = example_path(name)
    path_dict = ia.input_info_to_input_paths(folder_path)
    input_path = path_dict[kind + "_images_path"]
    input_dict = ia.input_info_to_input_dict(folder_path)
    path_dict = ia.input_info_to_output_paths(folder_path, input_dict)
    output_path = path_dict["segment_" + kind + "_path"]
    threshold_function_idx = 4
    zoom_function_idx = 2
    wound_name_list, tissue_name_list, contour_name_list, area_path, ax_maj_path, ax_min_path, tissue_path, is_broken_path, is_closed_path, _, _, _, _, _, _ = ia.run_segment(input_path, output_path, threshold_function_idx, zoom_function_idx)
    for wn in wound_name_list:
        assert wn.is_file()
    for tn in tissue_name_list:
        assert tn.is_file()
    for cn in contour_name_list:
        assert cn.is_file()
    assert area_path.is_file()
    assert ax_maj_path.is_file()
    assert ax_min_path.is_file()
    assert tissue_path.is_file()
    assert is_broken_path.is_file()
    assert is_closed_path.is_file()


def test_run_segment_bi():
    name = "test_ph1_mini_movie_bi"
    kind = "ph1"
    folder_path = example_path(name)
    path_dict = ia.input_info_to_input_paths(folder_path)
    input_path = path_dict[kind + "_images_path"]
    input_dict = ia.input_info_to_input_dict(folder_path)
    path_dict = ia.input_info_to_output_paths(folder_path, input_dict)
    output_path = path_dict["segment_" + kind + "_path"]
    threshold_function_idx = 4
    zoom_function_idx = 2
    _, tissue_name_list, _, _, _, _, tissue_path, is_broken_path, _, _, _, _, _, _, _ = ia.run_segment(input_path, output_path, threshold_function_idx, zoom_function_idx, is_bi=True)
    for tn in tissue_name_list:
        assert tn.is_file()
    assert tissue_path.is_file()
    assert is_broken_path.is_file()


def test_run_segment_dic():
    name = "test_dic_mini_movie"
    kind = "dic"
    folder_path = example_path(name)
    path_dict = ia.input_info_to_input_paths(folder_path)
    input_path = path_dict[kind + "_images_path"]
    input_dict = ia.input_info_to_input_dict(folder_path)
    path_dict = ia.input_info_to_output_paths(folder_path, input_dict)
    output_path = path_dict["segment_" + kind + "_path"]
    threshold_function_idx = 6
    zoom_function_idx = 2
    wound_name_list, tissue_name_list, contour_name_list, area_path, ax_maj_path, ax_min_path, tissue_path, is_broken_path, is_closed_path, _, _, _, _, _, _ = ia.run_segment(input_path, output_path, threshold_function_idx, zoom_function_idx)
    for wn in wound_name_list:
        assert wn.is_file()
    for tn in tissue_name_list:
        assert tn.is_file()
    for cn in contour_name_list:
        assert cn.is_file()
    assert area_path.is_file()
    assert ax_maj_path.is_file()
    assert ax_min_path.is_file()
    assert tissue_path.is_file()
    assert is_broken_path.is_file()
    assert is_closed_path.is_file()


def test_show_and_save_wound_area():
    num_frames = 50
    wound_area = np.random.rand(num_frames,)
    GPR_wound_area = np.random.rand(num_frames)
    output_path = files_path()

    ia.show_and_save_wound_area(
        wound_area,
        GPR_wound_area,
        output_path
    )
    output_file = output_path / "wound_area.png"
    assert output_file.exists()


def test_save_all_img_with_contour_and_create_gif_bf():
    for kind in ["brightfield", "fluorescent"]:
        zoom_fcn_idx = 1
        folder_path = example_path("test_mini_movie")
        path_dict = ia.input_info_to_input_paths(folder_path)
        input_dict = ia.input_info_to_input_dict(folder_path)
        input_path = path_dict[kind + "_images_path"]
        tiff_list = ia.read_all_tiff(input_path)
        if kind == "brightfield":
            img_list = copy.deepcopy(tiff_list)
        path_dict = ia.input_info_to_output_paths(folder_path, input_dict)
        output_path = path_dict["segment_" + kind + "_vis_path"]
        file_name = kind + "_contour"
        if kind == "brightfield":
            threshold_function_idx = 1
        if kind == "fluorescent":
            threshold_function_idx = 2
        thresholded_list = seg.threshold_all(tiff_list, threshold_function_idx)
        tissue_mask_list, wound_mask_list, wound_region_list = seg.mask_all(thresholded_list, 1)
        is_broken_list = com.check_broken_tissue_all(tissue_mask_list)
        is_closed_list = com.check_wound_closed_all(tissue_mask_list, wound_region_list, zoom_fcn_idx)
        contour_list = seg.contour_all(wound_mask_list)
        if kind == "brightfield":
            contour_list_bf = copy.deepcopy(contour_list)
        else:
            contour_list_fl = copy.deepcopy(contour_list)
        file_path = ia.save_all_img_with_contour(output_path, file_name, tiff_list, contour_list, is_broken_list, is_closed_list)
        assert len(file_path) == 5
        for file in file_path:
            assert file.is_file()
        gif_path = ia.create_gif(output_path, file_name, file_path)
        assert gif_path.is_file()
    output_path = path_dict["bf_seg_with_fl_seg_visualize_path"]
    file_name = "bf_with_fl"
    file_path = ia.save_all_img_with_double_contour(output_path, file_name, img_list, contour_list_bf, contour_list_fl)
    assert len(file_path) == 5
    for file in file_path:
        assert file.is_file()
    gif_path = ia.create_gif(output_path, file_name, file_path)
    assert gif_path.is_file()


def test_numpy_to_list():
    folder_path = example_path("test_io").joinpath("numpy_arrays").resolve()
    file_name = "test_save_numpy"
    file_list = ia.numpy_to_list(folder_path, file_name)
    assert len(file_list) == 2
    assert file_list[0].shape == (5, 5)
    assert file_list[1].shape == (5, 5)


def test_check_before_injury_folder():
    test_strings = [
    "/data/test_tissue_bi/S2/tissue_bi/s001_A02",  # True (two matches)
    "/data/test_tissue_bio/S2/tissue_biologic",     # False (has letters after _bi)
    "/data/test_bi_1",                              # True (_bi followed by a number)
    "/data/test_biologic",                          # False (_bio...)
    "/data/test_bi",                                # True (ends with _bi)
    "/data/test_bi/",                               # True (_bi followed by /)
    ]
    found = []
    for s in test_strings:
        found.append(ia.check_before_injury_folder(s))
    expected = [True, False, True, False, True, True]
    assert found == expected


def test_run_all():
    folder_path = example_path("test_mini_movie")
    time_all, action_all = ia.run_all(folder_path)
    assert len(time_all) == 7
    assert len(action_all) == 7


def test_run_all_ph1():
    folder_path = example_path("test_ph1_mini_movie")
    time_all, action_all = ia.run_all(folder_path)
    assert len(time_all) == 4
    assert len(action_all) == 4


def test_load_contour_coords():
    folder_path = example_path("test_ph1_mini_movie")
    _, _ = ia.run_all(folder_path)
    contour_coords_list = ia.load_contour_coords(folder_path,"ph1")
    assert len(contour_coords_list) > 0
    assert contour_coords_list[0].shape[0] > 0


def test_load_contour_coords_none():
    folder_path = example_path("test_zoom")
    contour_coords_list = ia.load_contour_coords(folder_path,"ph1")
    for contour in contour_coords_list:
        assert contour is None


def test_combine_images():

    dir = files_path().joinpath("test_combine_images/visualizations").resolve()
    image_type = "ph1"

    ia.combine_images(
        folder_path=dir,
        output_path=dir,
        image_type=image_type,
        max_combined_width=1000,
        individual_max_size=200
    )

    expected_output = dir / f"{image_type}_all_files.png"
    assert expected_output.exists()
    
    wrong_dir = files_path().joinpath("test_combine_images").resolve()
    ia.combine_images(
        folder_path=wrong_dir,
        output_path=wrong_dir,
        image_type="ph1",
        max_combined_width=1000,
        individual_max_size=200
    )


def test_run_all_ph1_broken():
    folder_path = example_path("test_ph1_mini_movie_broken")
    time_all, action_all = ia.run_all(folder_path)
    assert len(time_all) == 4
    assert len(action_all) == 4


def test_check_broken_tissue():
    tissue_mask = np.zeros((10, 10))
    tissue_mask[3:7, 3:7] = 1
    is_broken = com.check_broken_tissue(tissue_mask)
    assert is_broken is False
    tissue_mask[3:5, 3:5] = 0
    is_broken = com.check_broken_tissue(tissue_mask)
    assert is_broken is True
    tissue_mask = np.zeros((10, 10))
    is_broken = com.check_broken_tissue(tissue_mask)
    assert is_broken is True


def test_get_mean_center():
    array = np.zeros((10, 10))
    array[2:7, 4:6] = 1
    center_0, center_1 = seg.get_mean_center(array)
    assert center_0 == 4
    assert center_1 == 4.5


def test_run_all_closed():
    folder_path = example_path("test_mini_movie_closing")
    time_all, action_all = ia.run_all(folder_path)
    assert len(time_all) == 4
    assert len(action_all) == 4


def test_run_all_closed_ph1():
    folder_path = example_path("test_ph1_mini_movie_closed")
    time_all, action_all = ia.run_all(folder_path)
    assert len(time_all) == 4
    assert len(action_all) == 4


def test_run_all_ph1_many_examples():
    folder_path = example_path("test_phi_many_examples")
    time_all, action_all = ia.run_all(folder_path)
    assert len(time_all) == 4
    assert len(action_all) == 4


def test_run_all_ph1_anish():
    folder_path = example_path("test_ph1_movie_mini_Anish")
    time_all, action_all = ia.run_all(folder_path)
    assert len(time_all) == 5
    assert len(action_all) == 5


def test_run_all_ph1_many_examples_anish():
    folder_path = example_path("test_phi_many_examples_Anish")
    time_all, action_all = ia.run_all(folder_path)
    assert len(time_all) == 4
    assert len(action_all) == 4


def test_run_texture_tracking():
    folder_path = example_path("test_phi_movie_mini_Anish_tracking")
    input_path = folder_path.joinpath("ph1_images").resolve()
    output_path = ia.create_folder(folder_path, "track_ph1")
    threshold_function_idx = 4
    tracker_x_forward, tracker_y_forward, tracker_x_reverse_forward, tracker_y_reverse_forward, wound_area_list, wound_masks_all, path_tx, path_ty, path_txr, path_tyr, path_wound_area, path_wound_masks = ia.run_texture_tracking(input_path, output_path, threshold_function_idx)
    assert path_tx.is_file()
    assert path_ty.is_file()
    assert path_txr.is_file()
    assert path_tyr.is_file()
    assert path_wound_area.is_file()
    assert path_wound_masks.is_file()
    assert tracker_x_forward.shape[1] == tracker_y_forward.shape[1]
    assert tracker_x_reverse_forward.shape[1] == tracker_y_reverse_forward.shape[1]
    assert len(wound_area_list) == tracker_x_forward.shape[1]
    assert len(wound_masks_all) == tracker_x_forward.shape[1]


@pytest.mark.parametrize("n_plots,expected", [
    (0, (0, 0)),             # no plots
    (1, (1, 1)),             # single plot
    (2, (1, 2)),             # minimal non-square
    (4, (2, 2)),             # perfect square
    (6, (2, 3)),             # non-square, best-fit
    (7, (3, 3)),             # prime number
    (10, (3, 4)),            # fits in 3x4
    (12, (3, 4)),            # same as above
    (15, (4, 4)),            # close to square
    (100, (10, 10)),         # large square
    (101, (10, 11)),         # slightly over square
])
def test_get_subplot_dims(n_plots, expected):
    assert ia.get_subplot_dims(n_plots) == expected


def test_show_and_save_relative_pillar_distances():
    num_frames = 50
    num_pairs = 6
    relative_distances = np.random.rand(num_frames, num_pairs)
    GPR_relative_distances = np.random.rand(num_frames, num_pairs)
    rel_dist_pair_names = np.array([f"pair_{i}" for i in range(num_pairs)])
    output_path = files_path()

    ia.show_and_save_relative_pillar_distances(
        relative_distances,
        GPR_relative_distances,
        rel_dist_pair_names,
        output_path,
        True
    )

    output_file = output_path / "relative_pillar_distances.png"
    assert output_file.exists()


def test_show_and_save_pillar_positions():
    folder_path = example_path("test_phi_movie_mini_Anish_tracking")
    input_path = folder_path.joinpath("ph1_images").resolve()
    output_path = ia.create_folder(folder_path, "track_ph1")
    img_list = ia.read_all_tiff(input_path)
    pillar_mask_list,_ = seg.get_pillar_mask_list(img_list[0],num_pillars_expected=4,mask_seg_type=1)
    output_path = files_path()
    ia.show_and_save_pillar_positions(
        img_list[0],
        pillar_mask_list,
        output_path,
        title="Test Pillar Positions"
    )
    output_file = output_path / "pillar_positions.png"
    assert output_file.exists()


def test_show_and_save_pillar_disps_and_contours():
    folder_path = example_path("test_phi_movie_mini_Anish_tracking")
    input_path = folder_path.joinpath("ph1_images").resolve()
    output_path = ia.create_folder(folder_path, "track_ph1")
    img_list = ia.read_all_tiff(input_path)
    pillar_mask_list,_ = seg.get_pillar_mask_list(img_list[0],num_pillars_expected=4,mask_seg_type=1)
    pillars_pos_x,pillars_pos_y=tt.perform_pillar_tracking(pillar_mask_list,img_list,2)
    pillar_disps,avg_pillar_disps,_,_=pp.compute_absolute_actual_pillar_disps(pillars_pos_x,pillars_pos_y)
    ia.show_and_save_pillar_disps_and_contours(
        img_list[0],
        pillar_mask_list,
        pillar_disps,
        avg_pillar_disps,
        output_path
    )
    output_file = output_path / "pillar_disps_and_pillar_contours_test_phi_movie_mini_Anish_tracking.png"
    assert output_file.exists()


def test_run_texture_tracking_pillars():
    folder_path = example_path("test_phi_movie_mini_Anish_tracking")
    input_path = folder_path.joinpath("ph1_images").resolve()
    output_path = ia.create_folder(folder_path, "pillar_track_ph1")
    img_list = ia.read_all_tiff(input_path)
    # threshold_function_idx = 4
    pillars_mask_list,avg_disp_all_x, avg_disp_all_y, path_disp_x, path_disp_y = ia.run_texture_tracking_pillars(img_list, output_path, mask_seg_type=2)
    assert len(pillars_mask_list) == 4
    assert avg_disp_all_x.shape[0] == avg_disp_all_y.shape[0]
    assert path_disp_x.is_file()
    assert path_disp_y.is_file()


def test_run_texture_tracking_pillars_large_bg_shift():
    folder_path = example_path("test_ph1_movie_mini_large_bg_shift")
    input_path = folder_path.joinpath("ph1_images").resolve()
    output_path = ia.create_folder(folder_path, "pillar_track_ph1")
    img_list = ia.read_all_tiff(input_path)
    # threshold_function_idx = 4
    pillars_mask_list,avg_disp_all_x, avg_disp_all_y, path_disp_x, path_disp_y = ia.run_texture_tracking_pillars(img_list, output_path, mask_seg_type=2)
    assert len(pillars_mask_list) == 4
    assert avg_disp_all_x.shape[0] == avg_disp_all_y.shape[0]
    assert path_disp_x.is_file()
    assert path_disp_y.is_file()


def test_show_and_save_tracking():
    folder_path = example_path("test_phi_movie_mini_Anish_tracking")
    input_path = folder_path.joinpath("ph1_images").resolve()
    output_path = ia.create_folder(folder_path, "track_ph1")
    threshold_function_idx = 4
    tracker_x_forward, tracker_y_forward, tracker_x_reverse_forward, tracker_y_reverse_forward, wound_area_list, wound_masks_all, path_tx, path_ty, path_txr, path_tyr, path_wound_area, path_wound_masks = ia.run_texture_tracking(input_path, output_path, threshold_function_idx)
    img_list = ia.read_all_tiff(input_path)
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, wound_region_list = seg.mask_all(thresholded_list, 1)
    pillar_mask_list,_  = seg.get_pillar_mask_list(img_list[0],num_pillars_expected=4,mask_seg_type=1)
    is_broken_list = com.check_broken_tissue_all(tissue_mask_list,pillar_mask_list=pillar_mask_list)
    zoom_fcn_idx = 1
    is_closed_list = com.check_wound_closed_all(tissue_mask_list, wound_region_list, zoom_fcn_idx)
    contour_list = seg.contour_all(wound_mask_list)
    img = img_list[-1]
    contour = contour_list[-1]
    is_broken = is_broken_list[-1]
    is_closed = is_closed_list[-1]
    frame = len(img_list) - 1
    save_path = output_file("test_phi_movie_mini_Anish_tracking", "test_save_tracking_title.png")
    title = "example_anish_example"
    ia.show_and_save_tracking(img, contour, is_broken, is_closed, frame, tracker_x_forward, tracker_y_forward, tracker_x_reverse_forward, tracker_y_reverse_forward, save_path, title)
    assert save_path.is_file()
    is_broken = True
    ia.show_and_save_tracking(img, contour, is_broken, is_closed, frame, tracker_x_forward, tracker_y_forward, tracker_x_reverse_forward, tracker_y_reverse_forward, save_path, title)
    assert save_path.is_file()
    is_broken = False
    is_closed = True
    ia.show_and_save_tracking(img, contour, is_broken, is_closed, frame, tracker_x_forward, tracker_y_forward, tracker_x_reverse_forward, tracker_y_reverse_forward, save_path, title)
    assert save_path.is_file()


def test_save_all_img_tracking():
    folder_path = example_path("test_phi_movie_mini_Anish_tracking")
    input_path = folder_path.joinpath("ph1_images").resolve()
    output_path = ia.create_folder(folder_path, "track_ph1")
    threshold_function_idx = 4
    tracker_x_forward, tracker_y_forward, tracker_x_reverse_forward, tracker_y_reverse_forward, wound_area_list, wound_masks_all, path_tx, path_ty, path_txr, path_tyr, path_wound_area, path_wound_masks = ia.run_texture_tracking(input_path, output_path, threshold_function_idx)
    img_list = ia.read_all_tiff(input_path)
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, wound_region_list = seg.mask_all(thresholded_list, 1)
    is_broken_list = com.check_broken_tissue_all(tissue_mask_list)
    zoom_fcn_idx = 1
    is_closed_list = com.check_wound_closed_all(tissue_mask_list, wound_region_list, zoom_fcn_idx)
    contour_list = seg.contour_all(wound_mask_list)
    fname = "test"
    file_name_list = ia.save_all_img_tracking(output_path, fname, img_list, contour_list, is_broken_list, is_closed_list, tracker_x_forward, tracker_y_forward, tracker_x_reverse_forward, tracker_y_reverse_forward)
    for fn in file_name_list:
        assert fn.is_file()


def test_run_texture_tracking_visualize():
    folder_path = example_path("test_phi_movie_mini_Anish_tracking")
    input_path = folder_path.joinpath("ph1_images").resolve()
    output_path = ia.create_folder(folder_path, "track_ph1")
    threshold_function_idx = 4
    tracker_x_forward, tracker_y_forward, tracker_x_reverse_forward, tracker_y_reverse_forward, wound_area_list, wound_masks_all, path_tx, path_ty, path_txr, path_tyr, path_wound_area, path_wound_masks = ia.run_texture_tracking(input_path, output_path, threshold_function_idx)
    img_list = ia.read_all_tiff(input_path)
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, wound_region_list = seg.mask_all(thresholded_list, 1)
    is_broken_list = com.check_broken_tissue_all(tissue_mask_list)
    zoom_fcn_idx = 1
    is_closed_list = com.check_wound_closed_all(tissue_mask_list, wound_region_list, zoom_fcn_idx)
    contour_list = seg.contour_all(wound_mask_list)
    (path_list, gif_path) = ia.run_texture_tracking_visualize(output_path, img_list, contour_list, is_broken_list, is_closed_list, tracker_x_forward, tracker_y_forward, tracker_x_reverse_forward, tracker_y_reverse_forward)
    assert gif_path.is_file()
    for pa in path_list:
        assert pa.is_file()


def test_run_all_tracking():
    folder_path = example_path("test_phi_movie_mini_Anish_tracking")
    time_all, action_all = ia.run_all(folder_path)
    assert len(time_all) == 7
    assert len(action_all) == 7


def test_run_all_zoom_Anish():
    folder_path = example_path("test_NikonEclipse_Evolve512_10X")
    time_all, action_all = ia.run_all(folder_path)
    assert len(time_all) == 4
    assert len(action_all) == 4


def test_run_all_dic():
    folder_path = example_path("test_dic_mini_movie")
    time_all, action_all = ia.run_all(folder_path)
    assert len(time_all) == 5
    assert len(action_all) == 5


# def test_line_param():
#     array = np.zeros((50, 50))
#     array[5:40, 10:15] = 1
#     regions_all = seg.get_region_props(array)
#     region = seg.get_largest_regions(regions_all, 1)[0]
#     _, axis_major_length, _, centroid_row, centroid_col, _, _, orientation = seg.extract_region_props(region)
#     line_a, line_b, line_c = ia.line_param(centroid_row, centroid_col, orientation)
#     assert pytest.approx(line_a, 0.01) == 0.0
#     assert pytest.approx(line_b, 0.01) == 1.0
#     array = np.zeros((50, 50))
#     array[10:15, 5:40] = 1
#     regions_all = seg.get_region_props(array)
#     region = seg.get_largest_regions(regions_all, 1)[0]
#     _, axis_major_length, _, centroid_row, centroid_col, _, _, orientation = seg.extract_region_props(region)
#     line_a, line_b, line_c = ia.line_param(centroid_row, centroid_col, orientation)
#     assert np.abs(line_a) > 1.0 * 10 ** 10.0
#     assert pytest.approx(line_b, 0.01) == 1.0
#     array = np.zeros((50, 50))
#     array[10:15, 10:15] = 1
#     array[14:20, 14:20] = 1
#     regions_all = seg.get_region_props(array)
#     region = seg.get_largest_regions(regions_all, 1)[0]
#     _, axis_major_length, _, centroid_row, centroid_col, _, _, orientation = seg.extract_region_props(region)
#     line_a, line_b, line_c = ia.line_param(centroid_row, centroid_col, orientation)
#     assert pytest.approx(line_a, 0.01) == pytest.approx(line_b, 0.01)
#     # import matplotlib.pyplot as plt
#     # x = np.linspace(0, array.shape[0])
#     # y = -1.0 * line_a / line_b * x - line_c / line_b
#     # plt.imshow(array)
#     # plt.plot(x, y, 'r')
#     # aa = 44


# def test_dist_to_line():
#     array = np.zeros((50, 50))
#     array[5:40, 10:15] = 1
#     regions_all = seg.get_region_props(array)
#     region = seg.get_largest_regions(regions_all, 1)[0]
#     _, axis_major_length, _, centroid_row, centroid_col, _, _, orientation = seg.extract_region_props(region)
#     line_a, line_b, line_c = ia.line_param(centroid_row, centroid_col, orientation)
#     pt_0 = 0.0
#     pt_1 = 22.0
#     line_dist = ia.dist_to_line(line_a, line_b, line_c, pt_0, pt_1)
#     assert pytest.approx(line_dist, 0.01) == 0.0
#     array = np.zeros((50, 50))
#     array[10:15, 10:15] = 1
#     array[14:20, 14:20] = 1
#     regions_all = seg.get_region_props(array)
#     region = seg.get_largest_regions(regions_all, 1)[0]
#     _, axis_major_length, _, centroid_row, centroid_col, _, _, orientation = seg.extract_region_props(region)
#     line_a, line_b, line_c = ia.line_param(centroid_row, centroid_col, orientation)
#     pt_0 = 16.08088022903976
#     pt_1 = 16.08088022903976
#     line_dist = ia.dist_to_line(line_a, line_b, line_c, pt_0, pt_1)
#     assert pytest.approx(line_dist, 0.01) == 2.0
#     pt_0 = 13.25245310429357
#     pt_1 = 13.25245310429357
#     line_dist = ia.dist_to_line(line_a, line_b, line_c, pt_0, pt_1)
#     assert pytest.approx(line_dist, 0.01) == 2.0


# def test_clip_contour():
#     rad = 50
#     disk = morphology.disk(rad, bool)
#     array = np.zeros((rad * 4, rad * 4))
#     array[rad:rad + disk.shape[0], rad:rad + disk.shape[1]] = disk
#     contour = seg.mask_to_contour(array)
#     regions_all = seg.get_region_props(array)
#     region = seg.get_largest_regions(regions_all, 1)[0]
#     _, axis_major_length, axis_minor_length, centroid_row, centroid_col, _, _, orientation = seg.extract_region_props(region)
#     contour_clipped = ia.clip_contour(contour, centroid_row, centroid_col, orientation, axis_major_length * 2.0, axis_minor_length * 2.0)
#     assert np.allclose(contour, contour_clipped)
#     array = np.zeros((50, 50))
#     array[10:40, 20:25] = 1
#     contour = seg.mask_to_contour(array)
#     regions_all = seg.get_region_props(array)
#     region = seg.get_largest_regions(regions_all, 1)[0]
#     _, axis_major_length, axis_minor_length, centroid_row, centroid_col, _, _, orientation = seg.extract_region_props(region)
#     contour_clipped = ia.clip_contour(contour, centroid_row, centroid_col, orientation, axis_major_length, axis_minor_length)
#     # import matplotlib.pyplot as plt
#     # plt.plot(contour[:,0], contour[:,1])
#     # plt.plot(contour_clipped[:,0], contour_clipped[:,1],'r')
#     # plt.axis('equal')
#     assert np.allclose(contour_clipped, contour) is False


# def test_move_point():
#     pt_0 = -10
#     pt_1 = 10
#     line_a = 1
#     line_b = 1
#     line_c = 0
#     cutoff = 5
#     pt_0_new, pt_1_new = ia.move_point(pt_0, pt_1, line_a, line_b, line_c, cutoff)
#     assert pt_0_new == -10
#     assert pt_1_new == 10
#     pt_0 = -10
#     pt_1 = 10
#     line_a = -1
#     line_b = 1
#     line_c = 0
#     cutoff = 5 * np.sqrt(2)
#     pt_0_new, pt_1_new = ia.move_point(pt_0, pt_1, line_a, line_b, line_c, cutoff)
#     assert pytest.approx(pt_0_new, .01) == -5
#     assert pytest.approx(pt_1_new, .01) == 5
#     pt_0 = 10
#     pt_1 = -10
#     line_a = -1
#     line_b = 1
#     line_c = 0
#     cutoff = 5 * np.sqrt(2)
#     pt_0_new, pt_1_new = ia.move_point(pt_0, pt_1, line_a, line_b, line_c, cutoff)
#     assert pytest.approx(pt_0_new, .01) == 5
#     assert pytest.approx(pt_1_new, .01) == -5
#     pt_0 = 10
#     pt_1 = 10
#     line_a = 1
#     line_b = 1
#     line_c = 0
#     cutoff = 5 * np.sqrt(2)
#     pt_0_new, pt_1_new = ia.move_point(pt_0, pt_1, line_a, line_b, line_c, cutoff)
#     assert pytest.approx(pt_0_new, .01) == 5
#     assert pytest.approx(pt_1_new, .01) == 5
#     pt_0 = -10
#     pt_1 = -10
#     line_a = 1
#     line_b = 1
#     line_c = 0
#     cutoff = 5 * np.sqrt(2)
#     pt_0_new, pt_1_new = ia.move_point(pt_0, pt_1, line_a, line_b, line_c, cutoff)
#     assert pytest.approx(pt_0_new, .01) == -5
#     assert pytest.approx(pt_1_new, .01) == -5
#     pt_0 = 10
#     pt_1 = 10
#     line_a = 1
#     line_b = 0
#     line_c = 0
#     cutoff = 5
#     pt_0_new, pt_1_new = ia.move_point(pt_0, pt_1, line_a, line_b, line_c, cutoff)
#     assert pytest.approx(pt_0_new, .01) == 10
#     assert pytest.approx(pt_1_new, .01) == 5
#     pt_0 = 10
#     pt_1 = 10
#     line_a = 0
#     line_b = 1
#     line_c = 0
#     cutoff = 5
#     pt_0_new, pt_1_new = ia.move_point(pt_0, pt_1, line_a, line_b, line_c, cutoff)
#     assert pytest.approx(pt_0_new, .01) == 5
#     assert pytest.approx(pt_1_new, .01) == 10


# def test_resample_contour():
#     array = np.zeros((50, 50))
#     array[10:40, 20:25] = 1
#     contour = seg.mask_to_contour(array)
#     regions_all = seg.get_region_props(array)
#     region = seg.get_largest_regions(regions_all, 1)[0]
#     _, axis_major_length, axis_minor_length, centroid_row, centroid_col, _, _, orientation = seg.extract_region_props(region)
#     contour_clipped = ia.clip_contour(contour, centroid_row, centroid_col, orientation, axis_major_length, axis_minor_length)
#     resampled_contour = ia.resample_contour(contour_clipped)
#     assert resampled_contour.shape[0] == contour_clipped.shape[0]
#     array = np.zeros((1000, 1000))
#     array[200:800, 200:250] = 1
#     contour = seg.mask_to_contour(array)
#     regions_all = seg.get_region_props(array)
#     region = seg.get_largest_regions(regions_all, 1)[0]
#     _, axis_major_length, axis_minor_length, centroid_row, centroid_col, _, _, orientation = seg.extract_region_props(region)
#     contour_clipped = ia.clip_contour(contour, centroid_row, centroid_col, orientation, axis_major_length, axis_minor_length)
#     resampled_contour = ia.resample_contour(contour_clipped)
#     assert resampled_contour.shape[0] < contour_clipped.shape[0]
#     # make test more robust!
#     # import matplotlib.pyplot as plt
#     # plt.plot(contour[:,0], contour[:,1])
#     # plt.plot(contour_clipped[:,0], contour_clipped[:,1],'r-')
#     # plt.plot(resampled_contour[:,0], resampled_contour[:,1],'g.')
#     # plt.axis('equal')


# def test_get_penalized():
#     contour = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]])
#     contour_clipped = np.asarray([[1, 2], [3, 7], [8, 6], [9, 10]])
#     penalized_contour = ia.get_penalized(contour, contour_clipped)
#     assert penalized_contour[0, 0] == 1
#     assert penalized_contour[0, 1] == 2
#     for kk in range(1, 4):
#         assert penalized_contour[kk, 0] == math.inf
#         assert penalized_contour[kk, 1] == math.inf

