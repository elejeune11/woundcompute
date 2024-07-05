import glob
import numpy as np
from pathlib import Path
import pytest
from scipy import ndimage
from skimage import io
from skimage import morphology
from woundcompute import compute_values as com
from woundcompute import image_analysis as ia
from woundcompute import segmentation as seg


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


def test_apply_median_filter():
    array = np.ones((10, 10))
    array[1, 5] = 10
    array[7, 3] = 10
    filter_size = 3
    known = np.ones((10, 10))
    found = seg.apply_median_filter(array, filter_size)
    assert np.allclose(known, found)


def test_apply_gaussian_filter():
    array = np.ones((10, 10))
    array[1, 5] = 10
    array[7, 3] = 10
    filter_size = 3
    known = ndimage.gaussian_filter(array, filter_size)
    found = seg.apply_gaussian_filter(array, filter_size)
    assert np.allclose(known, found)


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
    found = seg.compute_otsu_thresh(x1)
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
    found = seg.apply_otsu_thresh(x1)
    assert np.allclose(known, found)


def test_apply_thresh_multiotsu():
    dim = 100
    arr = np.zeros((dim, dim))
    arr[0:10, :] = 1
    arr[50:70, 80:90] = 2
    known = arr > 0
    found = seg.apply_thresh_multiotsu(arr)
    assert np.allclose(known, found)


def test_get_region_props():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    rad_2 = 3
    disk_2 = morphology.disk(rad_2, dtype=bool)
    dim = 30
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    array[-disk_2.shape[0]:, -disk_2.shape[1]:] = disk_2
    region_props = seg.get_region_props(array)
    assert region_props[0].area == np.sum(disk_1)
    assert region_props[1].area == np.sum(disk_2)


def test_insert_borders():
    mask = np.ones((50, 50))
    border = 10
    mask = seg.insert_borders(mask, border)
    assert np.sum(mask) == 30 * 30


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
    region_props = seg.get_region_props(array)
    num_regions = 2
    regions_list = seg.get_largest_regions(region_props, num_regions)
    assert len(regions_list) == 2
    assert regions_list[0].area == np.sum(disk_1)
    assert regions_list[1].area == np.sum(disk_2)


def test_get_roundest_regions():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    dim = 30
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    array[-4:-2, 5:20] = 1
    array[15:17, 10:29] = 1
    region_props = seg.get_region_props(array)
    num_regions = 1
    regions_list = seg.get_roundest_regions(region_props, num_regions)
    assert len(regions_list) == 1
    assert regions_list[0].area == np.sum(disk_1)


def test_get_domain_center():
    array = np.ones((10, 20))
    center_0, center_1 = seg.get_domain_center(array)
    assert center_0 == 5
    assert center_1 == 10


def test_compute_distance():
    a0 = 0
    a1 = 0
    b0 = 10
    b1 = 0
    known = 10
    found = seg.compute_distance(a0, a1, b0, b1)
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
    region_props = seg.get_region_props(array)
    num_regions = 3
    regions_list = seg.get_largest_regions(region_props, num_regions)
    loc_0 = 0
    loc_1 = 0
    region = seg.get_closest_region(regions_list, loc_0, loc_1)
    assert region.area == np.sum(disk_1)


def test_extract_region_props():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    region_props = seg.get_region_props(disk_1)
    region = region_props[0]
    area, axis_major_length, axis_minor_length, centroid_row, centroid_col, coords, bbox, orientation = seg.extract_region_props(region)
    assert area == np.sum(disk_1)
    assert axis_major_length > 10
    assert axis_major_length < 11
    assert axis_minor_length > 10
    assert axis_minor_length < 11
    assert centroid_row == 5
    assert centroid_col == 5
    assert coords.shape[0] == np.sum(disk_1)
    assert bbox[0] == 0
    assert bbox[1] == 0
    assert bbox[2] == disk_1.shape[0]
    assert bbox[3] == disk_1.shape[1]
    assert orientation > 0
    region = None
    area, axis_major_length, axis_minor_length, centroid_row, centroid_col, coords, bbox, orientation = seg.extract_region_props(region)
    assert area is None
    assert axis_major_length is None
    assert axis_minor_length is None
    assert centroid_row is None
    assert centroid_col is None
    assert coords is None
    assert bbox[0] is None
    assert bbox[1] is None
    assert bbox[2] is None
    assert bbox[3] is None
    assert orientation is None


def test_get_regions_not_touching_bounds():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    rad_2 = 3
    disk_2 = morphology.disk(rad_2, dtype=bool)
    rad_3 = 2
    disk_3 = morphology.disk(rad_3, dtype=bool)
    dim = 30
    array = np.zeros((dim, dim))
    array[3:disk_1.shape[0] + 3, 3:disk_1.shape[1] + 3] = disk_1
    array[-disk_2.shape[0] - 3:-3, -disk_2.shape[1] - 3:-3] = disk_2
    array_new_2 = np.copy(array)
    array[0:disk_3.shape[0], -disk_3.shape[1]:] = disk_3
    region_props = seg.get_region_props(array)
    assert len(region_props) == 3
    region_props_new = seg.get_regions_not_touching_bounds(region_props, array.shape)
    assert len(region_props_new) == 2
    array_new = np.copy(array)
    array_new[-1, :] = 1
    region_props = seg.get_region_props(array_new)
    assert len(region_props) == 4
    region_props_new = seg.get_regions_not_touching_bounds(region_props, array_new.shape)
    assert len(region_props_new) == 2
    array_new = np.copy(array)
    array_new[:, 0] = 1
    region_props = seg.get_region_props(array_new)
    assert len(region_props) == 4
    region_props_new = seg.get_regions_not_touching_bounds(region_props, array_new.shape)
    assert len(region_props_new) == 2
    array_new_2[:, -1] = 1
    region_props = seg.get_region_props(array_new_2)
    assert len(region_props) == 3
    region_props_new = seg.get_regions_not_touching_bounds(region_props, array_new.shape)
    assert len(region_props_new) == 2


def test_check_above_min_size():
    array = np.zeros((20, 20))
    array[8:12, 8:12] = 1
    region = seg.get_region_props(array)[0]
    assert seg.check_above_min_size(region, 3) is True
    assert seg.check_above_min_size(region, 100) is False


def test_coords_to_mask():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    region_props = seg.get_region_props(disk_1)
    region = region_props[0]
    coords = [seg.extract_region_props(region)[5]]
    mask = seg.coords_to_mask(coords, disk_1)
    assert np.allclose(mask, disk_1)


def test_mask_to_contour():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    dim = 30
    array = np.zeros((dim, dim))
    ix = int(dim / 2)
    array[ix - rad_1:ix + rad_1 + 1, ix - rad_1:ix + rad_1 + 1] = disk_1
    contour = seg.mask_to_contour(array)
    assert np.mean(contour[:, 0]) > dim / 2 - 1
    assert np.mean(contour[:, 0]) < dim / 2 + 1
    assert np.mean(contour[:, 1]) > dim / 2 - 1
    assert np.mean(contour[:, 1]) < dim / 2 + 1
    array = np.zeros((dim, dim))
    contour = seg.mask_to_contour(array)
    assert contour is None


def test_invert_mask():
    array_half = np.zeros((10, 10))
    array_half[0:5, :] = 1
    array_invert = seg.invert_mask(array_half)
    assert np.allclose(array_invert + array_half, np.ones((10, 10)))


def test_coords_to_inverted_mask():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    region_props = seg.get_region_props(disk_1)
    region = region_props[0]
    coords = [seg.extract_region_props(region)[5]]
    mask = seg.coords_to_mask(coords, disk_1)
    mask_inverted = seg.coords_to_inverted_mask(coords, disk_1)
    assert np.allclose(mask + mask_inverted, np.ones(mask.shape))


def test_threshold_all():
    # brightfield use case
    file_path = glob_brightfield("test_single")[0]
    example_file = io.imread(file_path)
    thresh_img = seg.threshold_array(example_file, 1)
    assert np.max(thresh_img) == 1
    assert np.min(thresh_img) == 0
    assert thresh_img.shape[0] == example_file.shape[0]
    assert thresh_img.shape[1] == example_file.shape[1]
    # fluorescent use case
    file_path = glob_fluorescent("test_single")[0]
    example_file = io.imread(file_path)
    thresh_img = seg.threshold_array(example_file, 2)
    assert np.max(thresh_img) == 1
    assert np.min(thresh_img) == 0
    assert thresh_img.shape[0] == example_file.shape[0]
    assert thresh_img.shape[1] == example_file.shape[1]
    # ph1 use case
    file_path = glob_ph1("test_single")[0]
    example_file = io.imread(file_path)
    thresh_img = seg.threshold_array(example_file, 3)
    assert np.max(thresh_img) == 1
    assert np.min(thresh_img) == 0
    assert thresh_img.shape[0] == example_file.shape[0]
    assert thresh_img.shape[1] == example_file.shape[1]
    # ph1 use case -- Anish data
    file_path = glob_ph1("test_single")[1]
    example_file = io.imread(file_path)
    thresh_img = seg.threshold_array(example_file, 4)
    assert np.max(thresh_img) == 1
    assert np.min(thresh_img) == 0
    assert thresh_img.shape[0] == example_file.shape[0]
    assert thresh_img.shape[1] == example_file.shape[1]
    # spectra use case -- fluorescent
    file_path = glob_fluorescent("test_single")[1]
    example_file = io.imread(file_path)
    thresh_img = seg.threshold_array(example_file, 5)
    assert np.max(thresh_img) == 1
    assert np.min(thresh_img) == 0
    assert thresh_img.shape[0] == example_file.shape[0]
    assert thresh_img.shape[1] == example_file.shape[1]
    # error due to unaccounted for case
    with pytest.raises(ValueError) as error:
        seg.threshold_array(example_file, 15)
    assert error.typename == "ValueError"


def test_gabor_filter():
    array = np.zeros((10, 10))
    gabor_all = seg.gabor_filter(array)
    assert np.sum(gabor_all) == 0
    file_path = glob_ph1("test_single")[1]
    img = io.imread(file_path)
    gabor_all = seg.gabor_filter(img)
    assert np.allclose(gabor_all, img) is False
    theta_range = 9
    ff_max = 6
    ff_mult = 0.05
    gabor_all_2 = seg.gabor_filter(img, theta_range, ff_max, ff_mult)
    assert np.allclose(gabor_all_2, img) is False


def test_preview_thresholding():
    file_path = glob_ph1("test_single")[1]
    img = io.imread(file_path)
    thresh_list, idx_list = seg.preview_thresholding(img)
    assert len(thresh_list) == 4
    assert len(idx_list) == 4
    assert np.allclose(thresh_list[0].shape, img.shape)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(img)
    # plt.figure()
    # for kk in range(0, 4):
    #     plt.subplot(2, 2, kk + 1)
    #     plt.imshow(thresh_list[kk])
    #     plt.title(str(kk + 1))
    # aa = 44


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
    region_props = seg.get_region_props(array)
    coords_list = seg.region_to_coords(region_props)
    assert len(coords_list) == 3
    assert coords_list[0].shape[1] == 2


def test_isolate_masks_gfp():
    file_path = glob_fluorescent("test_single")[0]
    example_file = io.imread(file_path)
    thresh_img = seg.threshold_array(example_file, 2)
    tissue_mask, wound_mask, _ = seg.isolate_masks(thresh_img, 2)
    assert np.max(tissue_mask) == 1
    assert np.min(tissue_mask) == 0
    assert tissue_mask.shape[0] == tissue_mask.shape[0]
    assert tissue_mask.shape[1] == tissue_mask.shape[1]
    assert np.max(wound_mask) == 1
    assert np.min(wound_mask) == 0
    assert wound_mask.shape[0] == wound_mask.shape[0]
    assert wound_mask.shape[1] == wound_mask.shape[1]
    assert np.sum(wound_mask + tissue_mask <= 1) == wound_mask.shape[0] * wound_mask.shape[1]
    file_path = glob_fluorescent("test_single")[1]
    example_file = io.imread(file_path)
    thresh_img = seg.threshold_array(example_file, 5)
    tissue_mask, wound_mask, _ = seg.isolate_masks(thresh_img, 5)
    assert np.max(tissue_mask) == 1
    assert np.min(tissue_mask) == 0
    assert tissue_mask.shape[0] == tissue_mask.shape[0]
    assert tissue_mask.shape[1] == tissue_mask.shape[1]
    assert np.max(wound_mask) == 1
    assert np.min(wound_mask) == 0
    assert wound_mask.shape[0] == wound_mask.shape[0]
    assert wound_mask.shape[1] == wound_mask.shape[1]
    assert np.sum(wound_mask + tissue_mask <= 1) == wound_mask.shape[0] * wound_mask.shape[1]


def test_isolate_masks_brightfield():
    file_path = glob_brightfield("test_single")[0]
    example_file = io.imread(file_path)
    thresh_img = seg.threshold_array(example_file, 1)
    tissue_mask, wound_mask, _ = seg.isolate_masks(thresh_img, 1)
    assert np.max(tissue_mask) == 1
    assert np.min(tissue_mask) == 0
    assert tissue_mask.shape[0] == tissue_mask.shape[0]
    assert tissue_mask.shape[1] == tissue_mask.shape[1]
    assert np.max(wound_mask) == 1
    assert np.min(wound_mask) == 0
    assert wound_mask.shape[0] == wound_mask.shape[0]
    assert wound_mask.shape[1] == wound_mask.shape[1]
    assert np.sum(wound_mask + tissue_mask <= 1) == wound_mask.shape[0] * wound_mask.shape[1]


# def test_isolate_masks_zoom_examples():
#     folder_path = example_path("test_zoom")
#     input_dict, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
#     folder_path = input_path_dict["ph1_images_path"]
#     img_list = ia.read_all_tiff(folder_path)
#     path_list = ia.image_folder_to_path_list(folder_path)
#     threshold_function_idx = seg.select_threshold_function(input_dict, False, False, True)
#     thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
#     tissue_mask_list, wound_mask_list, _ = seg.mask_all(thresholded_list, threshold_function_idx)
#     import matplotlib.pyplot as plt
#     for kk in range(0, len(img_list)):
#         fig, ax = plt.subplots(1, 3)
#         ax[0].imshow(img_list[kk])
#         ax[1].imshow(tissue_mask_list[kk])
#         ax[2].imshow(wound_mask_list[kk])
#         ti = str(path_list[kk]).split("/")[-1]
#         ax[0].set_title(ti)
#         ax[1].set_title(ti)
#     plt.figure()
#     plt.imshow(img_list[8])
#     plt.figure()
#     plt.imshow(img_list[13])
#     aa = 44


def test_isolate_masks_no_wound():
    thresh_img = np.ones((100, 100))
    thresh_img[30:50, :] = 0
    tissue_mask, wound_mask, wound_region = seg.isolate_masks(thresh_img, 1)
    assert np.allclose(thresh_img < 1, tissue_mask)
    assert np.allclose(wound_mask, np.zeros((100, 100)))
    assert wound_region is None
    is_closed = com.check_wound_closed(tissue_mask, wound_region)
    assert is_closed


def test_isolate_masks_ph1():
    file_path = glob_ph1("test_single")[0]
    example_file = io.imread(file_path)
    thresh_img = seg.threshold_array(example_file, 3)
    tissue_mask, wound_mask, _ = seg.isolate_masks(thresh_img, 3)
    assert np.max(tissue_mask) == 1
    assert np.min(tissue_mask) == 0
    assert tissue_mask.shape[0] == tissue_mask.shape[0]
    assert tissue_mask.shape[1] == tissue_mask.shape[1]
    assert np.max(wound_mask) == 1
    assert np.min(wound_mask) == 0
    assert wound_mask.shape[0] == wound_mask.shape[0]
    assert wound_mask.shape[1] == wound_mask.shape[1]
    assert np.sum(wound_mask + tissue_mask <= 1) == wound_mask.shape[0] * wound_mask.shape[1]


def test_isolate_masks_ph1_anish():
    file_path = glob_ph1("test_single")[1]
    example_file = io.imread(file_path)
    thresh_img = seg.threshold_array(example_file, 4)
    tissue_mask, wound_mask, _ = seg.isolate_masks(thresh_img, 4)
    assert np.max(tissue_mask) == 1
    assert np.min(tissue_mask) == 0
    assert tissue_mask.shape[0] == tissue_mask.shape[0]
    assert tissue_mask.shape[1] == tissue_mask.shape[1]
    assert np.max(wound_mask) == 1
    assert np.min(wound_mask) == 0
    assert wound_mask.shape[0] == wound_mask.shape[0]
    assert wound_mask.shape[1] == wound_mask.shape[1]
    assert np.sum(wound_mask + tissue_mask <= 1) == wound_mask.shape[0] * wound_mask.shape[1]


def test_isolate_masks_other_case():
    # error due to unaccounted for case
    with pytest.raises(ValueError) as error:
        seg.isolate_masks(np.zeros((10, 10)), 15)
    assert error.typename == "ValueError"


def test_close_region():
    val = 10
    array = np.zeros((val, val))
    array[3:7, 3:7] = 1
    array_missing = np.copy(array)
    array_missing[5, 5] = 0
    array_closed = seg.close_region(array_missing)
    assert np.allclose(array_closed, array)


def test_dilate_region():
    val = 10
    array = np.zeros((val, val))
    array[3:7, 3:7] = 1
    array_dilated = seg.dilate_region(array) * 1.0
    assert np.sum(array_dilated) == 6 * 6 - 4


def test_select_threshold_function():
    folder_path = example_path("test_single")
    input_dict = ia.input_info_to_input_dict(folder_path)
    is_brightfield = True
    is_fluorescent = False
    is_ph1 = False
    assert seg.select_threshold_function(input_dict, is_brightfield, is_fluorescent, is_ph1) == 1
    is_brightfield = False
    is_fluorescent = True
    is_ph1 = False
    assert seg.select_threshold_function(input_dict, is_brightfield, is_fluorescent, is_ph1) == 2
    is_brightfield = False
    is_fluorescent = False
    is_ph1 = True
    assert seg.select_threshold_function(input_dict, is_brightfield, is_fluorescent, is_ph1) == 3

    input_dict["seg_ph1_version"] = 2
    is_brightfield = False
    is_fluorescent = False
    is_ph1 = True
    assert seg.select_threshold_function(input_dict, is_brightfield, is_fluorescent, is_ph1) == 4

    input_dict["seg_bf_version"] = 2
    is_brightfield = True
    is_fluorescent = False
    is_ph1 = False
    with pytest.raises(ValueError) as error:
        seg.select_threshold_function(input_dict, is_brightfield, is_fluorescent, is_ph1)
    assert error.typename == "ValueError"

    input_dict["seg_fl_version"] = 2
    is_brightfield = False
    is_fluorescent = True
    is_ph1 = False
    assert seg.select_threshold_function(input_dict, is_brightfield, is_fluorescent, is_ph1) == 5

    input_dict["seg_fl_version"] = 3
    is_brightfield = False
    is_fluorescent = True
    is_ph1 = False
    with pytest.raises(ValueError) as error:
        seg.select_threshold_function(input_dict, is_brightfield, is_fluorescent, is_ph1)
    assert error.typename == "ValueError"

    input_dict["seg_ph1_version"] = 3
    is_brightfield = False
    is_fluorescent = False
    is_ph1 = True
    with pytest.raises(ValueError) as error:
        seg.select_threshold_function(input_dict, is_brightfield, is_fluorescent, is_ph1)
    assert error.typename == "ValueError"


def test_single_masks_ph1_special_cases():
    file_path = glob_ph1("test_single_fail")[0]
    example_file = io.imread(file_path)
    thresh_img = seg.threshold_array(example_file, 4)
    tissue_mask, wound_mask, wound_region = seg.isolate_masks(thresh_img, 4)
    is_broken = com.check_broken_tissue(tissue_mask)
    is_closed = com.check_wound_closed(tissue_mask, wound_region)
    assert np.max(tissue_mask) == 1
    assert np.min(tissue_mask) == 0
    assert tissue_mask.shape[0] == tissue_mask.shape[0]
    assert tissue_mask.shape[1] == tissue_mask.shape[1]
    assert np.max(wound_mask) == 1
    assert np.min(wound_mask) == 0
    assert wound_mask.shape[0] == wound_mask.shape[0]
    assert wound_mask.shape[1] == wound_mask.shape[1]
    assert np.sum(wound_mask + tissue_mask <= 1) == wound_mask.shape[0] * wound_mask.shape[1]
    assert is_broken is False
    assert is_closed is False


def test_fill_tissue_mask_reconstruction():
    mask = np.zeros((50, 50))
    mask[14:34, 14:34] = 1
    mask_rect = np.copy(mask)
    mask[20:30, 20:30] = 0
    new_mask = seg.fill_tissue_mask_reconstruction(mask)
    mask_rect[10:38, 10:38] = 1
    assert np.allclose(new_mask, mask_rect)


def test_make_tissue_mask_robust():
    folder_path = example_path("test_ph1_movie_mini_Anish")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    img_list = [img_list[0]]
    threshold_function_idx = 4
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, _ = seg.mask_all(thresholded_list, threshold_function_idx)
    tissue_mask = tissue_mask_list[0]
    wound_mask = wound_mask_list[0]
    tissue_mask_robust = seg.make_tissue_mask_robust(tissue_mask, wound_mask)
    tissue_contour = seg.mask_to_contour(tissue_mask_robust)
    assert tissue_contour.shape[0] > 100


def test_make_tissue_mask_robust_brightfield():
    folder_path = example_path("test_mini_movie")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["brightfield_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    img_list = [img_list[0]]
    threshold_function_idx = 4
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, _ = seg.mask_all(thresholded_list, threshold_function_idx)
    tissue_mask = tissue_mask_list[0]
    wound_mask = wound_mask_list[0]
    tissue_mask_robust = seg.make_tissue_mask_robust(tissue_mask, wound_mask)
    tissue_contour = seg.mask_to_contour(tissue_mask_robust)
    assert tissue_contour.shape[0] > 100


def test_get_pillar_mask_list():
    folder_path = example_path("test_pillar_tracking")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    img_list = [img_list[0]]
    threshold_function_idx = 4
    pillar_mask_list = seg.get_pillar_mask_list(img_list[0], threshold_function_idx)
    assert len(pillar_mask_list) == 4
    for kk in range(0, 4):
        assert np.sum(pillar_mask_list[kk]) > 10
        assert np.min(pillar_mask_list[kk]) == 0
        assert np.max(pillar_mask_list[kk]) == 1
    # test an example where only 3 pillars are picked up
    img_list = ia.read_all_tiff(folder_path)
    img_list = [img_list[-1]]
    threshold_function_idx = 4
    mask_seg_type = 2
    pillar_mask_list = seg.get_pillar_mask_list(img_list[0], threshold_function_idx, mask_seg_type)
    assert len(pillar_mask_list) == 3


# def test_mask_quadrants_img():
#     frame = np.random.random((100, 200))
#     quadrant = 0
#     masked_image = seg.mask_quadrants_img(frame, quadrant)
#     assert masked_image[10, 10] > 0
#     assert masked_image[90, 150] == 0
#     quadrant = 1
#     masked_image = seg.mask_quadrants_img(frame, quadrant)
#     assert masked_image[80, 50] > 0
#     assert masked_image[20, 150] == 0
#     quadrant = 2
#     masked_image = seg.mask_quadrants_img(frame, quadrant)
#     assert masked_image[30, 150] > 0
#     assert masked_image[90, 50] == 0
#     quadrant = 3
#     masked_image = seg.mask_quadrants_img(frame, quadrant)
#     assert masked_image[80, 150] > 0
#     assert masked_image[10, 10] == 0

def test_mask_img_for_pillar_track():
    ix0 = 100
    ix1 = 200
    img = (np.random.random((ix0, ix1)) * 255).astype("uint8")
    pillar_mask = np.zeros((ix0, ix1))
    rad_1 = 3
    disk_1 = morphology.disk(rad_1, dtype=bool)
    pillar_mask[50:57, 100:107] = disk_1
    masked_img = seg.mask_img_for_pillar_track(img, pillar_mask, 10)
    assert np.allclose(pillar_mask * masked_img, pillar_mask * img)
    assert masked_img[20, 75] == 0
    assert masked_img[80, 100] == 0
    assert masked_img[60, 175] == 0


def test_pillar_mask_to_box():
    ix0 = 200
    ix1 = 300
    img = np.random.random((ix0, ix1))
    pillar_mask = np.zeros((ix0, ix1))
    rad_1 = 3
    disk_1 = morphology.disk(rad_1, dtype=bool)
    mix_0_min = 50
    mix_0_max = 57
    mix_1_min = 100
    mix_1_max = 107
    pillar_mask[mix_0_min:mix_0_max, mix_1_min:mix_1_max] = disk_1
    buffer = 10
    r_min, r_max, c_min, c_max = seg.pillar_mask_to_box(img, pillar_mask, buffer)
    assert r_min == mix_0_min - buffer
    assert r_max == mix_0_max + buffer
    assert c_min == mix_1_min - buffer
    assert c_max == mix_1_max + buffer


def test_mask_to_template():
    ix0 = 200
    ix1 = 300
    img = np.random.random((ix0, ix1))
    pillar_mask = np.zeros((ix0, ix1))
    rad_1 = 3
    disk_1 = morphology.disk(rad_1, dtype=bool)
    mix_0_min = 50
    mix_0_max = 57
    mix_1_min = 100
    mix_1_max = 107
    pillar_mask[mix_0_min:mix_0_max, mix_1_min:mix_1_max] = disk_1
    buffer = 0
    template = seg.mask_to_template(img, pillar_mask, buffer)
    assert template.shape == disk_1.shape
    template = seg.mask_to_template(pillar_mask, pillar_mask, buffer)
    assert np.allclose(template, disk_1)


def test_uint16_to_uint8():
    arr = (np.random.random((10, 10)) * 1000 + 1).astype("uint16")
    arr[4, 4] = 1200
    arr[3, 3] = 0
    arr_8 = seg.uint16_to_uint8(arr)
    assert arr_8.dtype is np.dtype("uint8")
    assert np.min(arr_8) == 0
    assert np.max(arr_8) == 255


def test_thresh_img_local():
    arr = np.zeros((500, 500))
    arr[50:100, 50:100] = 1000
    arr[400:450, 400:450] = 100
    gt_mask = (arr > 0).astype("uint8")
    mask = seg.thresh_img_local(arr).astype("uint8")
    assert np.allclose(gt_mask, mask)


def test_contour_to_mask():
    mask = np.zeros((100, 100))
    mask[50:90, 60:80] = 1
    contour = seg.mask_to_contour(mask)
    mask_new = seg.contour_to_mask(mask, contour)
    assert mask_new.shape == mask.shape
    assert np.isclose(np.sum(mask_new), np.sum(mask), 100)
    mask = np.zeros((100, 100))
    contour = seg.mask_to_contour(mask)
    mask_new = seg.contour_to_mask(mask, contour)
    assert mask_new.shape == mask.shape
    assert np.allclose(mask, mask_new)
    assert contour is None


def test_contour_to_region():
    mask = np.zeros((100, 100))
    mask[50:90, 60:80] = 1
    contour = seg.mask_to_contour(mask)
    mask_new = seg.contour_to_mask(mask, contour)
    wound_region = seg.contour_to_region(mask, contour)
    assert np.sum(mask_new) == wound_region.area
    mask = np.zeros((100, 100))
    contour = seg.mask_to_contour(mask)
    wound_region = seg.contour_to_region(mask, contour)
    assert wound_region is None


# def test_sequence_tissue_segment():
#     folder_path = example_path("test_ph1_movie_mini_Anish")
#     _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
#     folder_path = input_path_dict["ph1_images_path"]
#     img_list = ia.read_all_tiff(folder_path)
#     img_n = img_list[1]
#     img_np1 = img_list[2]
#     seg.sequence_tissue_segment(img_n, img_np1)


def test_pillar_mask_to_rotated_box():
    mask = np.zeros((1000, 1000))
    mask[250:300, 420:450] = 1
    mask[250:300, 520:550] = 1
    mask[700:750, 420:450] = 1
    mask[700:750, 520:550] = 1
    box = seg.pillar_mask_to_rotated_box(mask)
    assert box.shape == (4, 2)
    assert np.isclose(np.min(box[:, 0]), 250, atol=3)
    assert np.isclose(np.max(box[:, 0]), 750, atol=3)
    assert np.isclose(np.min(box[:, 1]), 420, atol=3)
    assert np.isclose(np.max(box[:, 1]), 550, atol=3)


def test_compute_unit_vector():
    x1 = 0
    x2 = 10
    y1 = 0
    y2 = 0
    vec = seg.compute_unit_vector(x1, x2, y1, y2)
    assert np.allclose(vec, np.asarray([1, 0]))
    x1 = 0
    x2 = 10
    y1 = 0
    y2 = 10
    vec = seg.compute_unit_vector(x1, x2, y1, y2)
    assert np.allclose(vec, np.asarray([np.sqrt(2) / 2.0, np.sqrt(2) / 2.0]))


def test_box_to_unit_vec_len_wid():
    box = np.asarray([[0, 0], [0, 10], [5, 10], [5, 0]])
    vec, leng, wid = seg.box_to_unit_vec_len_wid(box)
    assert np.allclose(vec, np.asarray([0, 1]), atol=.1) or np.allclose(vec, np.asarray([0, -1]), atol=.1)
    assert np.isclose(leng, 10.0)
    assert np.isclose(wid, 5.0)
    box = np.asarray([[0, 0], [0, 5], [10, 5], [10, 0]])
    vec, leng, wid = seg.box_to_unit_vec_len_wid(box)
    assert np.allclose(vec, np.asarray([1, 0])) or np.allclose(vec, np.asarray([-1, 0]))
    assert np.isclose(leng, 10.0)
    assert np.isclose(wid, 5.0)


def test_box_to_center_points():
    box = np.asarray([[0, 0], [0, 10], [5, 10], [5, 0]])
    center_row, center_col = seg.box_to_center_points(box)
    assert np.isclose(center_row, 2.5)
    assert np.isclose(center_col, 5.0)


def test_mask_list_to_single_mask():
    mask_list = []
    for kk in range(0, 5):
        ma = np.zeros((100, 100))
        ma[kk*10:(kk + 3)*10, kk*10:(kk + 3)*10] = 1
        mask_list.append(ma)
    mask = seg.mask_list_to_single_mask(mask_list)
    assert mask.shape == (100, 100)
    assert np.max(mask) == 1
    assert np.min(mask) == 0


def test_move_point_closer():
    pt_0 = 0
    pt_1 = 0
    c_0 = 10
    c_1 = 10
    scale_factor = 0.5
    new_pt_0, new_pt_1 = seg.move_point_closer(pt_0, pt_1, c_0, c_1, scale_factor)
    assert np.isclose(new_pt_0, 5.0)
    assert np.isclose(new_pt_1, 5.0)
    pt_0 = 10
    pt_1 = 0
    c_0 = 3
    c_1 = 4
    scale_factor = 1.0
    new_pt_0, new_pt_1 = seg.move_point_closer(pt_0, pt_1, c_0, c_1, scale_factor)
    assert np.isclose(new_pt_0, c_0)
    assert np.isclose(new_pt_1, c_1)


def test_shrink_box():
    box = np.asarray([[0, 0], [0, 10], [5, 10], [5, 0]])
    scale_factor = 0.5
    scale_box = seg.shrink_box(box, scale_factor)
    assert np.isclose(scale_box[0, 0], 5.0 / 4.0)
    assert np.isclose(scale_box[2, 0], 5.0 - 5.0 / 4.0)
    assert np.isclose(scale_box[1, 1], 10.0 - 10.0 / 4.0)
    assert np.isclose(scale_box[3, 1], 10.0 / 4.0)
    ang = np.pi / 4.0
    R = np.asarray([[np.cos(ang), np.sin(ang) * - 1.0], [np.sin(ang), np.cos(ang)]])
    box_new = []
    for kk in range(0, 4):
        vec = np.dot(R, box[kk, :])
        box_new.append(vec)
    box_new = np.asarray(box_new)
    scale_box = seg.shrink_box(box_new, scale_factor)
    assert np.isclose(scale_box[0, 0], -0.88388348)
    assert np.isclose(scale_box[2, 0], -2.651650429449553)
    assert np.isclose(scale_box[1, 1], 6.187184335382291)
    assert np.isclose(scale_box[3, 1], 4.419417382415922)


def test_area_triangle_3_pts():
    x0 = 0
    y0 = 0
    x1 = 10
    y1 = 0
    x2 = 10
    y2 = 10
    area = seg.area_triangle_3_pts(x0, x1, x2, y0, y1, y2)
    assert np.isclose(area, 50.0)
    x0 = 0
    y0 = 0
    x1 = 10
    y1 = 0
    x2 = 5
    y2 = 10
    area = seg.area_triangle_3_pts(x0, x1, x2, y0, y1, y2)
    assert np.isclose(area, 50.0)


def test_point_in_box():
    box = np.asarray([[0, 0], [0, 10], [5, 10], [5, 0]])
    pt_0 = 0.5
    pt_1 = 5.0
    in_box = seg.point_in_box(box, pt_0, pt_1)
    assert in_box is True
    pt_0 = -5
    pt_1 = 5
    in_box = seg.point_in_box(box, pt_0, pt_1)
    assert in_box is False


def test_regions_in_box():
    folder_path = example_path("test_pillar_tracking")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    img = img_list[0]
    threshold_function_idx = 4
    pillar_mask_list = seg.get_pillar_mask_list(img, threshold_function_idx)
    pillar_mask = seg.mask_list_to_single_mask(pillar_mask_list)
    pillar_box = seg.pillar_mask_to_rotated_box(pillar_mask)
    thresh_img = seg.threshold_array(img, 4)
    _, _, wound_region = seg.isolate_masks(thresh_img, 4)
    region_in_box = seg.regions_in_box(pillar_box, wound_region)
    assert region_in_box is True
    scale_factor = 0.95
    pillar_box_shrink = seg.shrink_box(pillar_box, scale_factor)
    region_in_box = seg.regions_in_box(pillar_box_shrink, wound_region)
    assert region_in_box is False
    # test regions_in_box_all
    wound_region_list = [wound_region, wound_region, wound_region]
    regions_keep = seg.regions_in_box_all(pillar_box, wound_region_list)
    assert len(regions_keep) == 3
    regions_keep = seg.regions_in_box_all(pillar_box_shrink, wound_region_list)
    assert len(regions_keep) == 0


def test_leverage_pillars_for_wound_seg():
    folder_path = example_path("test_phi_movie_mini_Anish_tracking")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    img = img_list[0]
    threshold_function_idx = 4
    pillar_mask_list = seg.get_pillar_mask_list(img, threshold_function_idx)
    pillar_mask = seg.mask_list_to_single_mask(pillar_mask_list)
    background_mask = seg.threshold_array(img, threshold_function_idx)
    tissue_mask, wound_mask, wound_region = seg.leverage_pillars_for_wound_seg(pillar_mask, background_mask)
    assert tissue_mask.shape == img.shape
    assert wound_mask.shape == img.shape
    assert wound_region is not None


def test_mask_all_with_pillars():
    folder_path = example_path("test_phi_movie_mini_Anish_tracking")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = 4
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    pillar_mask_list = seg.get_pillar_mask_list(img_list[0], threshold_function_idx)
    tissue_mask_list, wound_mask_list, wound_region_list = seg.mask_all_with_pillars(thresholded_list, pillar_mask_list)
    assert len(tissue_mask_list) == len(img_list)
    assert len(wound_mask_list) == len(img_list)
    assert len(wound_region_list) == len(img_list)
    assert tissue_mask_list[0].shape == img_list[0].shape
    assert wound_mask_list[0].shape == img_list[0].shape
