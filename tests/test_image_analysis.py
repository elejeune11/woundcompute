import glob
import numpy as np
from pathlib import Path
import pytest
from scipy import ndimage
from skimage import io
from skimage import morphology
from woundcompute import image_analysis as ia


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


def test_apply_median_filter():
    array = np.ones((10, 10))
    array[1, 5] = 10
    array[7, 3] = 10
    filter_size = 3
    known = np.ones((10, 10))
    found = ia.apply_median_filter(array, filter_size)
    assert np.all(known == found)


def test_apply_gaussian_filter():
    array = np.ones((10, 10))
    array[1, 5] = 10
    array[7, 3] = 10
    filter_size = 3
    known = ndimage.gaussian_filter(array, filter_size)
    found = ia.apply_gaussian_filter(array, filter_size)
    assert np.all(known == found)


def test_compute_otsu_thresh():
    dim = 10
    known_lower = 10
    known_upper = 100
    std_lower = 2
    std_upper = 10
    select = 0.8
    x1 = np.random.normal(known_lower, std_lower, dim * dim * dim)
    x1 = np.reshape(x1, (dim, dim, dim))
    x2 = np.random.normal(known_upper, std_upper, dim * dim * dim)
    x2 = np.reshape(x2, (dim, dim, dim))
    choose = np.random.random((dim, dim, dim)) > select
    x1[choose] = x1[choose] + x2[choose]
    found = ia.compute_otsu_thresh(x1)
    assert found > known_lower and found < (known_upper + known_lower)


def test_apply_otsu_thresh():
    dim = 10
    known_lower = 10
    known_upper = 10000
    std_lower = 0.1
    std_upper = 10
    select = 0.8
    x1 = np.random.normal(known_lower, std_lower, dim * dim * dim)
    x1 = np.reshape(x1, (dim, dim, dim))
    x2 = np.random.normal(known_upper, std_upper, dim * dim * dim)
    x2 = np.reshape(x2, (dim, dim, dim))
    choose = np.random.random((dim, dim, dim)) > select
    x1[choose] = x1[choose] + x2[choose]
    known = x1 > np.mean(x1)
    found = ia.apply_otsu_thresh(x1)
    assert np.all(known == found)


def test_get_region_props():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    rad_2 = 3
    disk_2 = morphology.disk(rad_2, dtype=bool)
    dim = 30
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    array[-disk_2.shape[0]:, -disk_2.shape[1]:] = disk_2
    region_props = ia.get_region_props(array)
    assert region_props[0].area == np.sum(disk_1)
    assert region_props[1].area == np.sum(disk_2)


def test_get_largest_regions():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    rad_2 = 3
    disk_2 = morphology.disk(rad_2, dtype=bool)
    rad_3 = 2
    disk_3 = morphology.disk(rad_3, dtype=bool)
    dim = 30
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    array[-disk_2.shape[0]:, -disk_2.shape[1]:] = disk_2
    array[0:disk_3.shape[0], -disk_3.shape[1]:] = disk_3
    region_props = ia.get_region_props(array)
    num_regions = 2
    regions_list = ia.get_largest_regions(region_props, num_regions)
    assert len(regions_list) == 2
    assert regions_list[0].area == np.sum(disk_1)
    assert regions_list[1].area == np.sum(disk_2)


def test_get_domain_center():
    array = np.ones((10, 20))
    center_0, center_1 = ia.get_domain_center(array)
    assert center_0 == 5
    assert center_1 == 10


def test_compute_distance():
    a0 = 0
    a1 = 0
    b0 = 10
    b1 = 0
    known = 10
    found = ia.compute_distance(a0, a1, b0, b1)
    assert known == found


def test_get_closest_region():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    rad_2 = 3
    disk_2 = morphology.disk(rad_2, dtype=bool)
    rad_3 = 2
    disk_3 = morphology.disk(rad_3, dtype=bool)
    dim = 30
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    array[-disk_2.shape[0]:, -disk_2.shape[1]:] = disk_2
    array[0:disk_3.shape[0], -disk_3.shape[1]:] = disk_3
    region_props = ia.get_region_props(array)
    num_regions = 3
    regions_list = ia.get_largest_regions(region_props, num_regions)
    loc_0 = 0
    loc_1 = 0
    region = ia.get_closest_region(regions_list, loc_0, loc_1)
    assert region.area == np.sum(disk_1)


def test_extract_region_props():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    region_props = ia.get_region_props(disk_1)
    region = region_props[0]
    area, axis_major_length, axis_minor_length, centroid_row, centroid_col, coords = ia.extract_region_props(region)
    assert area == np.sum(disk_1)
    assert axis_major_length > 10
    assert axis_major_length < 11
    assert axis_minor_length > 10
    assert axis_minor_length < 11
    assert centroid_row == 5
    assert centroid_col == 5
    assert coords.shape[0] == np.sum(disk_1)


def test_coords_to_mask():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    region_props = ia.get_region_props(disk_1)
    region = region_props[0]
    coords = [ia.extract_region_props(region)[5]]
    mask = ia.coords_to_mask(coords, disk_1)
    assert np.all(mask == disk_1)


def test_mask_to_contour():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    dim = 30
    array = np.zeros((dim, dim))
    ix = int(dim / 2)
    array[ix - rad_1:ix + rad_1 + 1, ix - rad_1:ix + rad_1 + 1] = disk_1
    contour = ia.mask_to_contour(array)
    assert np.mean(contour[:, 0]) > dim / 2 - 1
    assert np.mean(contour[:, 0]) < dim / 2 + 1
    assert np.mean(contour[:, 1]) > dim / 2 - 1
    assert np.mean(contour[:, 1]) < dim / 2 + 1


def test_invert_mask():
    array_half = np.zeros((10, 10))
    array_half[0:5, :] = 1
    array_invert = ia.invert_mask(array_half)
    assert np.all(array_invert + array_half == np.ones((10, 10)))


def test_coords_to_inverted_mask():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    region_props = ia.get_region_props(disk_1)
    region = region_props[0]
    coords = [ia.extract_region_props(region)[5]]
    mask = ia.coords_to_mask(coords, disk_1)
    mask_inverted = ia.coords_to_inverted_mask(coords, disk_1)
    assert np.all(mask + mask_inverted == np.ones(mask.shape))


def test_mask_to_area():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    region_props = ia.get_region_props(disk_1)
    region = region_props[0]
    coords = [ia.extract_region_props(region)[5]]
    mask = ia.coords_to_mask(coords, disk_1)
    pix_to_microns = 1
    area = ia.mask_to_area(mask, pix_to_microns)
    assert area == np.sum(disk_1)


def test_threshold_all():
    # brightfield use case
    file_path = glob_brightfield("test_single")[0]
    example_file = io.imread(file_path)
    thresh_img = ia.threshold_array(example_file, 1)
    assert np.max(thresh_img) == 1
    assert np.min(thresh_img) == 0
    assert thresh_img.shape[0] == example_file.shape[0]
    assert thresh_img.shape[1] == example_file.shape[1]
    # fluorescent use case
    file_path = glob_fluorescent("test_single")[0]
    example_file = io.imread(file_path)
    thresh_img = ia.threshold_array(example_file, 2)
    assert np.max(thresh_img) == 1
    assert np.min(thresh_img) == 0
    assert thresh_img.shape[0] == example_file.shape[0]
    assert thresh_img.shape[1] == example_file.shape[1]
    # error due to unaccounted for case
    with pytest.raises(ValueError) as error:
        ia.threshold_array(example_file, 3)
    assert error.typename == "ValueError"


def test_region_to_coords():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    rad_2 = 3
    disk_2 = morphology.disk(rad_2, dtype=bool)
    rad_3 = 2
    disk_3 = morphology.disk(rad_3, dtype=bool)
    dim = 30
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    array[-disk_2.shape[0]:, -disk_2.shape[1]:] = disk_2
    array[0:disk_3.shape[0], -disk_3.shape[1]:] = disk_3
    region_props = ia.get_region_props(array)
    coords_list = ia.region_to_coords(region_props)
    assert len(coords_list) == 3
    assert coords_list[0].shape[1] == 2


def test_isolate_masks_gfp():
    file_path = glob_fluorescent("test_single")[0]
    example_file = io.imread(file_path)
    thresh_img = ia.threshold_array(example_file, 2)
    tissue_mask, wound_mask, wound_region = ia.isolate_masks(thresh_img)
    assert np.max(tissue_mask) == 1
    assert np.min(tissue_mask) == 0
    assert tissue_mask.shape[0] == tissue_mask.shape[0]
    assert tissue_mask.shape[1] == tissue_mask.shape[1]
    assert np.max(wound_mask) == 1
    assert np.min(wound_mask) == 0
    assert wound_mask.shape[0] == wound_mask.shape[0]
    assert wound_mask.shape[1] == wound_mask.shape[1]
    assert np.all(wound_mask + tissue_mask <= 1)


def test_isolate_masks_brightfield():
    file_path = glob_brightfield("test_single")[0]
    example_file = io.imread(file_path)
    thresh_img = ia.threshold_array(example_file, 1)
    tissue_mask, wound_mask, wound_region = ia.isolate_masks(thresh_img)
    assert np.max(tissue_mask) == 1
    assert np.min(tissue_mask) == 0
    assert tissue_mask.shape[0] == tissue_mask.shape[0]
    assert tissue_mask.shape[1] == tissue_mask.shape[1]
    assert np.max(wound_mask) == 1
    assert np.min(wound_mask) == 0
    assert wound_mask.shape[0] == wound_mask.shape[0]
    assert wound_mask.shape[1] == wound_mask.shape[1]
    assert np.all(wound_mask + tissue_mask <= 1)


def test_read_tiff():
    file_path = glob_brightfield("test_single")[0]
    known = io.imread(file_path)
    found = ia.read_tiff(file_path)
    assert np.all(known == found)


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


def test_show_and_save_image_mask():
    file_path = glob_brightfield("test_single")[0]
    file = ia.read_tiff(file_path)
    file_thresh = ia.threshold_array(file, 1)
    tissue_mask, wound_mask, wound_region = ia.isolate_masks(file_thresh)
    save_path = output_file("test_single", "test_brightfield_tissue_mask.png")
    ia.show_and_save_image(tissue_mask, save_path)
    assert save_path.is_file()
    save_path = output_file("test_single", "test_brightfield_wound_mask.png")
    ia.show_and_save_image(wound_mask, save_path)
    assert save_path.is_file()
    file_path = glob_fluorescent("test_single")[0]
    file = ia.read_tiff(file_path)
    file_thresh = ia.threshold_array(file, 2)
    tissue_mask, wound_mask, wound_region = ia.isolate_masks(file_thresh)
    save_path = output_file("test_single", "test_gfp_tissue_mask.png")
    ia.show_and_save_image(tissue_mask, save_path)
    assert save_path.is_file()
    save_path = output_file("test_single", "test_gfp_wound_mask.png")
    ia.show_and_save_image(wound_mask, save_path)
    assert save_path.is_file()


def test_show_and_save_contour():
    file_path = glob_brightfield("test_single")[0]
    file = ia.read_tiff(file_path)
    file_thresh = ia.threshold_array(file, 1)
    wound_mask = ia.isolate_masks(file_thresh)[1]
    contour = ia.mask_to_contour(wound_mask)
    save_path = output_file("test_single", "test_brightfield_wound_contour.png")
    ia.show_and_save_contour(file, contour, save_path)
    assert save_path.is_file()


def test_save_numpy():
    data_path = files_path()
    file_path = glob_brightfield("test_single")[0]
    file = ia.read_tiff(file_path)
    save_path = data_path.joinpath("test_brightfield_save_no_title.npy")
    ia.save_numpy(file, save_path)
    assert save_path.is_file()
    file_thresh = ia.threshold_array(file, 1)
    tissue_mask, wound_mask, wound_region = ia.isolate_masks(file_thresh)
    save_path = data_path.joinpath("test_brightfield_tissue_mask.npy")
    ia.save_numpy(tissue_mask, save_path)
    assert save_path.is_file()
    contour = ia.mask_to_contour(wound_mask)
    save_path = data_path.joinpath("test_brightfield_tissue_contour.npy")
    ia.save_numpy(contour, save_path)
    assert save_path.is_file()


# def test_save_yaml():
#     rad_1 = 5
#     disk_1 = morphology.disk(rad_1, dtype=bool)
#     region_props = ia.get_region_props(disk_1)
#     region = region_props[0]
#     area, axis_major_length, axis_minor_length, centroid_row, centroid_col, coords = ia.extract_region_props(region)
#     data_path = files_path()
#     file_path = data_path.joinpath("test_save.yaml")
#     ia.save_yaml(area, axis_major_length, axis_minor_length, centroid_row, centroid_col, file_path)
#     assert file_path.is_file()


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


def test_create_folder():
    folder_path = example_path("test_io")
    new_folder_name = "test_create_folder"
    new_folder = ia.create_folder(folder_path, new_folder_name)
    assert new_folder.is_dir()


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


def test_select_threshold_function():
    folder_path = example_path("test_single")
    input_dict = ia.input_info_to_input_dict(folder_path)
    is_brightfield = True
    assert ia.select_threshold_function(input_dict, is_brightfield) == 1
    is_brightfield = False
    assert ia.select_threshold_function(input_dict, is_brightfield) == 2

    input_dict["seg_bf_version"] = 2
    is_brightfield = True
    with pytest.raises(ValueError) as error:
        ia.select_threshold_function(input_dict, is_brightfield)
    assert error.typename == "ValueError"

    input_dict["seg_fl_version"] = 2
    is_brightfield = False
    with pytest.raises(ValueError) as error:
        ia.select_threshold_function(input_dict, is_brightfield)
    assert error.typename == "ValueError"


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
    file_name_list = ia.save_all_numpy(folder_path, file_name, array_list)
    for file_name in file_name_list:
        assert file_name.is_file()


def test_save_list():
    folder_path = example_path("test_io")
    file_name = "test_save_list"
    value_list = [1, 2, 3, 4, 5]
    saved_name = ia.save_list(folder_path, file_name, value_list)
    assert saved_name.is_file()


def test_thresh_all():
    folder_path = example_path("test_mini_movie")
    path_dict = ia.input_info_to_input_paths(folder_path)
    folder_path = path_dict["brightfield_images_path"]
    tiff_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = 1
    thresholded_list = ia.threshold_all(tiff_list, threshold_function_idx)
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
    thresholded_list = ia.threshold_all(tiff_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, wound_region_list = ia.mask_all(thresholded_list)
    assert len(tissue_mask_list) == 5
    assert len(wound_mask_list) == 5
    assert len(wound_region_list) == 5
    for img in tissue_mask_list:
        assert np.max(img) == 1
        assert np.min(img) == 0
    for img in wound_mask_list:
        assert np.max(img) == 1
        assert np.min(img) == 0


def test_contour_all_and_parameters_all():
    folder_path = example_path("test_mini_movie")
    path_dict = ia.input_info_to_input_paths(folder_path)
    folder_path = path_dict["brightfield_images_path"]
    tiff_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = 1
    thresholded_list = ia.threshold_all(tiff_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, wound_region_list = ia.mask_all(thresholded_list)
    contour_list = ia.contour_all(wound_mask_list)
    area_list, axis_major_length_list, axis_minor_length_list = ia.parameters_all(wound_region_list)
    assert len(tissue_mask_list) == 5
    assert len(contour_list) == 5
    assert len(wound_mask_list) == 5
    assert len(area_list) == 5
    assert len(axis_major_length_list) == 5
    assert len(axis_minor_length_list) == 5
    assert np.max(area_list) < 512 * 512
    assert np.min(area_list) >= 0
    assert np.max(axis_major_length_list) < 512
    assert np.min(axis_major_length_list) >= 0
    assert np.max(axis_minor_length_list) < 512
    assert np.min(axis_minor_length_list) >= 0
    for kk in range(0, 5):
        assert axis_major_length_list[kk] >= axis_minor_length_list[kk]


def test_run_segment():
    for name in ["test_single", "test_mini_movie"]:
        for kind in ["brightfield", "fluorescent"]:
            folder_path = example_path(name)
            path_dict = ia.input_info_to_input_paths(folder_path)
            input_path = path_dict[kind + "_images_path"]
            input_dict = ia.input_info_to_input_dict(folder_path)
            path_dict = ia.input_info_to_output_paths(folder_path, input_dict)
            output_path = path_dict["segment_" + kind + "_path"]
            threshold_function_idx = 1
            wound_name_list, tissue_name_list, contour_name_list, area_path, ax_maj_path, ax_min_path = ia.run_segment(input_path, output_path, threshold_function_idx)
            for wn in wound_name_list:
                assert wn.is_file()
            for tn in tissue_name_list:
                assert tn.is_file()
            for cn in contour_name_list:
                assert cn.is_file()
            assert area_path.is_file()
            assert ax_maj_path.is_file()
            assert ax_min_path.is_file()
