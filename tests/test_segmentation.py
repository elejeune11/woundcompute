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