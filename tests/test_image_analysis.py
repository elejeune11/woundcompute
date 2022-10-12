import copy
import glob
import math
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


def test_apply_median_filter():
    array = np.ones((10, 10))
    array[1, 5] = 10
    array[7, 3] = 10
    filter_size = 3
    known = np.ones((10, 10))
    found = ia.apply_median_filter(array, filter_size)
    assert np.allclose(known, found)


def test_apply_gaussian_filter():
    array = np.ones((10, 10))
    array[1, 5] = 10
    array[7, 3] = 10
    filter_size = 3
    known = ndimage.gaussian_filter(array, filter_size)
    found = ia.apply_gaussian_filter(array, filter_size)
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


def test_get_roundest_regions():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    dim = 30
    array = np.zeros((dim, dim))
    array[0:disk_1.shape[0], 0:disk_1.shape[1]] = disk_1
    array[-4:-2, 5:20] = 1
    array[15:17, 10:29] = 1
    region_props = ia.get_region_props(array)
    num_regions = 1
    regions_list = ia.get_roundest_regions(region_props, num_regions)
    assert len(regions_list) == 1
    assert regions_list[0].area == np.sum(disk_1)


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
    area, axis_major_length, axis_minor_length, centroid_row, centroid_col, coords, bbox, orientation = ia.extract_region_props(region)
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
    area, axis_major_length, axis_minor_length, centroid_row, centroid_col, coords, bbox, orientation = ia.extract_region_props(region)
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
    region_props = ia.get_region_props(array)
    assert len(region_props) == 3
    region_props_new = ia.get_regions_not_touching_bounds(region_props, array.shape)
    assert len(region_props_new) == 2
    array_new = np.copy(array)
    array_new[-1, :] = 1
    region_props = ia.get_region_props(array_new)
    assert len(region_props) == 4
    region_props_new = ia.get_regions_not_touching_bounds(region_props, array_new.shape)
    assert len(region_props_new) == 2
    array_new = np.copy(array)
    array_new[:, 0] = 1
    region_props = ia.get_region_props(array_new)
    assert len(region_props) == 4
    region_props_new = ia.get_regions_not_touching_bounds(region_props, array_new.shape)
    assert len(region_props_new) == 2
    array_new_2[:, -1] = 1
    region_props = ia.get_region_props(array_new_2)
    assert len(region_props) == 3
    region_props_new = ia.get_regions_not_touching_bounds(region_props, array_new.shape)
    assert len(region_props_new) == 2


def test_shrink_bounding_box():
    (min_row, min_col, max_row, max_col) = (100, 50, 140, 130)
    shrink_factor = 0.5
    (min_row_new, min_col_new, max_row_new, max_col_new) = ia.shrink_bounding_box(min_row, min_col, max_row, max_col, shrink_factor)
    assert min_row_new == 100 + 10
    assert min_col_new == 50 + 20
    assert max_row_new == 140 - 10
    assert max_col_new == 130 - 20


def test_check_inside_box():
    array = np.zeros((20, 20))
    array[8:12, 8:12] = 1
    region = ia.get_region_props(array)[0]
    bbox = (5, 5, 15, 15)
    assert ia.check_inside_box(region, bbox, bbox) is True
    bbox = (5, 10, 15, 11)
    assert ia.check_inside_box(region, bbox, bbox) is False


def test_check_above_min_size():
    array = np.zeros((20, 20))
    array[8:12, 8:12] = 1
    region = ia.get_region_props(array)[0]
    assert ia.check_above_min_size(region, 3) is True
    assert ia.check_above_min_size(region, 100) is False


def test_coords_to_mask():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    region_props = ia.get_region_props(disk_1)
    region = region_props[0]
    coords = [ia.extract_region_props(region)[5]]
    mask = ia.coords_to_mask(coords, disk_1)
    assert np.allclose(mask, disk_1)


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
    array = np.zeros((dim, dim))
    contour = ia.mask_to_contour(array)
    assert contour is None


def test_invert_mask():
    array_half = np.zeros((10, 10))
    array_half[0:5, :] = 1
    array_invert = ia.invert_mask(array_half)
    assert np.allclose(array_invert + array_half, np.ones((10, 10)))


def test_coords_to_inverted_mask():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    region_props = ia.get_region_props(disk_1)
    region = region_props[0]
    coords = [ia.extract_region_props(region)[5]]
    mask = ia.coords_to_mask(coords, disk_1)
    mask_inverted = ia.coords_to_inverted_mask(coords, disk_1)
    assert np.allclose(mask + mask_inverted, np.ones(mask.shape))


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
    # ph1 use case
    file_path = glob_ph1("test_single")[0]
    example_file = io.imread(file_path)
    thresh_img = ia.threshold_array(example_file, 3)
    assert np.max(thresh_img) == 1
    assert np.min(thresh_img) == 0
    assert thresh_img.shape[0] == example_file.shape[0]
    assert thresh_img.shape[1] == example_file.shape[1]
    # ph1 use case -- Anish data
    file_path = glob_ph1("test_single")[1]
    example_file = io.imread(file_path)
    thresh_img = ia.threshold_array(example_file, 4)
    assert np.max(thresh_img) == 1
    assert np.min(thresh_img) == 0
    assert thresh_img.shape[0] == example_file.shape[0]
    assert thresh_img.shape[1] == example_file.shape[1]
    # error due to unaccounted for case
    with pytest.raises(ValueError) as error:
        ia.threshold_array(example_file, 15)
    assert error.typename == "ValueError"


def test_gabor_filter():
    array = np.zeros((10, 10))
    gabor_all = ia.gabor_filter(array)
    assert np.sum(gabor_all) == 0
    file_path = glob_ph1("test_single")[1]
    img = io.imread(file_path)
    gabor_all = ia.gabor_filter(img)
    assert np.allclose(gabor_all, img) is False
    theta_range = 9
    ff_max = 6
    ff_mult = 0.05
    gabor_all_2 = ia.gabor_filter(img, theta_range, ff_max, ff_mult)
    assert np.allclose(gabor_all_2, img) is False


def test_preview_thresholding():
    file_path = glob_ph1("test_single")[1]
    img = io.imread(file_path)
    thresh_list, idx_list = ia.preview_thresholding(img)
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
    region_props = ia.get_region_props(array)
    coords_list = ia.region_to_coords(region_props)
    assert len(coords_list) == 3
    assert coords_list[0].shape[1] == 2


def test_isolate_masks_gfp():
    file_path = glob_fluorescent("test_single")[0]
    example_file = io.imread(file_path)
    thresh_img = ia.threshold_array(example_file, 2)
    tissue_mask, wound_mask, wound_region = ia.isolate_masks(thresh_img, 2)
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
    thresh_img = ia.threshold_array(example_file, 1)
    tissue_mask, wound_mask, wound_region = ia.isolate_masks(thresh_img, 1)
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
    tissue_mask, wound_mask, wound_region = ia.isolate_masks(thresh_img, 1)
    assert np.allclose(thresh_img < 1, tissue_mask)
    assert np.allclose(wound_mask, np.zeros((100, 100)))
    assert wound_region is None
    is_closed = ia.check_wound_closed(tissue_mask, wound_region)
    assert is_closed


def test_isolate_masks_ph1():
    file_path = glob_ph1("test_single")[0]
    example_file = io.imread(file_path)
    thresh_img = ia.threshold_array(example_file, 3)
    tissue_mask, wound_mask, wound_region = ia.isolate_masks(thresh_img, 3)
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
    thresh_img = ia.threshold_array(example_file, 4)
    tissue_mask, wound_mask, wound_region = ia.isolate_masks(thresh_img, 4)
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
        ia.isolate_masks(np.zeros((10, 10)), 15)
    assert error.typename == "ValueError"


def test_read_tiff():
    file_path = glob_brightfield("test_single")[0]
    known = io.imread(file_path)
    found = ia.read_tiff(file_path)
    assert np.allclose(known, found)


def test_uint16_to_uint8():
    array_8 = np.random.randint(0, 255, (5, 5)).astype(np.uint8)
    array_8[0, 0] = 0
    array_8[1, 0] = 255
    array_16 = array_8.astype(np.uint16) * 100
    found = ia.uint16_to_uint8(array_16)
    assert np.allclose(array_8, found)


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


def test_close_region():
    val = 10
    array = np.zeros((val, val))
    array[3:7, 3:7] = 1
    array_missing = np.copy(array)
    array_missing[5, 5] = 0
    array_closed = ia.close_region(array_missing)
    assert np.allclose(array_closed, array)


def test_dilate_region():
    val = 10
    array = np.zeros((val, val))
    array[3:7, 3:7] = 1
    array_dilated = ia.dilate_region(array) * 1.0
    assert np.sum(array_dilated) == 6 * 6 - 4


def test_show_and_save_image_mask_bf():
    file_path = glob_brightfield("test_single")[0]
    file = ia.read_tiff(file_path)
    file_thresh = ia.threshold_array(file, 1)
    tissue_mask, wound_mask, wound_region = ia.isolate_masks(file_thresh, 1)
    save_path = output_file("test_single", "test_brightfield_tissue_mask.png")
    ia.show_and_save_image(tissue_mask, save_path)
    assert save_path.is_file()
    save_path = output_file("test_single", "test_brightfield_wound_mask.png")
    ia.show_and_save_image(wound_mask, save_path)
    assert save_path.is_file()


def test_show_and_save_image_mask_fl():
    file_path = glob_fluorescent("test_single")[0]
    file = ia.read_tiff(file_path)
    file_thresh = ia.threshold_array(file, 2)
    tissue_mask, wound_mask, wound_region = ia.isolate_masks(file_thresh, 2)
    save_path = output_file("test_single", "test_gfp_tissue_mask.png")
    ia.show_and_save_image(tissue_mask, save_path)
    assert save_path.is_file()
    save_path = output_file("test_single", "test_gfp_wound_mask.png")
    ia.show_and_save_image(wound_mask, save_path)
    assert save_path.is_file()


def test_show_and_save_image_mask_ph1():
    file_path = glob_ph1("test_single")[0]
    file = ia.read_tiff(file_path)
    file_thresh = ia.threshold_array(file, 3)
    save_path = output_file("test_single", "test_ph1_orig_mask.png")
    ia.show_and_save_image(file_thresh, save_path)
    tissue_mask, wound_mask, wound_region = ia.isolate_masks(file_thresh, 3)
    save_path = output_file("test_single", "test_ph1_tissue_mask.png")
    ia.show_and_save_image(tissue_mask, save_path)
    assert save_path.is_file()
    save_path = output_file("test_single", "test_ph1_wound_mask.png")
    ia.show_and_save_image(wound_mask, save_path)
    assert save_path.is_file()


def test_show_and_save_contour():
    file_path = glob_brightfield("test_single")[0]
    file = ia.read_tiff(file_path)
    file_thresh = ia.threshold_array(file, 1)
    wound_mask = ia.isolate_masks(file_thresh, 1)[1]
    contour = ia.mask_to_contour(wound_mask)
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


def test_show_and_save_contour_and_width():
    file_path = glob_brightfield("test_single")[0]
    file = ia.read_tiff(file_path)
    selection_idx = 1
    file_thresh = ia.threshold_array(file, selection_idx)
    wound_mask = ia.isolate_masks(file_thresh, selection_idx)[1]
    contour = ia.mask_to_contour(wound_mask)
    save_path = output_file("test_single", "test_brightfield_wound_contour.png")
    is_broken = False
    is_closed = False
    thresholded_list = [file_thresh]
    tissue_mask_list, wound_mask_list, _ = ia.mask_all(thresholded_list, selection_idx)
    tissue_parameters = ia.tissue_parameters_all([tissue_mask_list[0]], [wound_mask_list[0]])[0]
    points = [[tissue_parameters[1], tissue_parameters[3]], [tissue_parameters[2], tissue_parameters[4]]]
    ia.show_and_save_contour_and_width(file, contour, is_broken, is_closed, points, save_path)
    assert save_path.is_file()
    save_path = output_file("test_single", "test_brightfield_wound_contour_broken_label.png")
    is_broken = True
    is_closed = False
    ia.show_and_save_contour_and_width(file, contour, is_broken, is_closed, points, save_path)
    assert save_path.is_file()
    save_path = output_file("test_single", "test_brightfield_wound_contour_closed_label.png")
    is_broken = False
    is_closed = True
    ia.show_and_save_contour_and_width(file, contour, is_broken, is_closed, points, save_path)
    assert save_path.is_file()
    save_path = output_file("test_single", "test_brightfield_wound_contour_no_labels.png")
    is_broken = False
    is_closed = False
    contour = None
    points = None
    ia.show_and_save_contour_and_width(file, contour, is_broken, is_closed, points, save_path)
    assert save_path.is_file()


def test_show_and_save_double_contour():
    file_path = glob_brightfield("test_single")[0]
    file = ia.read_tiff(file_path)
    file_thresh = ia.threshold_array(file, 1)
    wound_mask = ia.isolate_masks(file_thresh, 1)[1]
    contour_bf = ia.mask_to_contour(wound_mask)
    file_path = glob_fluorescent("test_single")[0]
    file = ia.read_tiff(file_path)
    file_thresh = ia.threshold_array(file, 2)
    wound_mask = ia.isolate_masks(file_thresh, 2)[1]
    contour_fl = ia.mask_to_contour(wound_mask)
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
    file_thresh = ia.threshold_array(file, 1)
    tissue_mask, wound_mask, wound_region = ia.isolate_masks(file_thresh, 1)
    save_path = data_path.joinpath("test_brightfield_tissue_mask.npy")
    ia.save_numpy(tissue_mask, save_path)
    assert save_path.is_file()
    contour = ia.mask_to_contour(wound_mask)
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
    is_fluorescent = False
    is_ph1 = False
    assert ia.select_threshold_function(input_dict, is_brightfield, is_fluorescent, is_ph1) == 1
    is_brightfield = False
    is_fluorescent = True
    is_ph1 = False
    assert ia.select_threshold_function(input_dict, is_brightfield, is_fluorescent, is_ph1) == 2
    is_brightfield = False
    is_fluorescent = False
    is_ph1 = True
    assert ia.select_threshold_function(input_dict, is_brightfield, is_fluorescent, is_ph1) == 3

    input_dict["seg_ph1_version"] = 2
    is_brightfield = False
    is_fluorescent = False
    is_ph1 = True
    assert ia.select_threshold_function(input_dict, is_brightfield, is_fluorescent, is_ph1) == 4

    input_dict["seg_bf_version"] = 2
    is_brightfield = True
    is_fluorescent = False
    is_ph1 = False
    with pytest.raises(ValueError) as error:
        ia.select_threshold_function(input_dict, is_brightfield, is_fluorescent, is_ph1)
    assert error.typename == "ValueError"

    input_dict["seg_fl_version"] = 2
    is_brightfield = False
    is_fluorescent = True
    is_ph1 = False
    with pytest.raises(ValueError) as error:
        ia.select_threshold_function(input_dict, is_brightfield, is_fluorescent, is_ph1)
    assert error.typename == "ValueError"

    input_dict["seg_ph1_version"] = 3
    is_brightfield = False
    is_fluorescent = False
    is_ph1 = True
    with pytest.raises(ValueError) as error:
        ia.select_threshold_function(input_dict, is_brightfield, is_fluorescent, is_ph1)
    assert error.typename == "ValueError"


def test_read_all_tiff():
    folder_path = example_path("test_mini_movie")
    path_dict = ia.input_info_to_input_paths(folder_path)
    folder_path = path_dict["brightfield_images_path"]
    tiff_list = ia.read_all_tiff(folder_path)
    assert len(tiff_list) == 5
    assert tiff_list[0].shape == (512, 512)


def test_uint16_to_uint8_all():
    folder_path = example_path("test_mini_movie")
    path_dict = ia.input_info_to_input_paths(folder_path)
    folder_path = path_dict["brightfield_images_path"]
    tiff_list = ia.read_all_tiff(folder_path)
    uint8_list = ia.uint16_to_uint8_all(tiff_list)
    for img in uint8_list:
        assert img.dtype is np.dtype('uint8')


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
    tissue_mask_list, wound_mask_list, wound_region_list = ia.mask_all(thresholded_list, 1)
    assert len(tissue_mask_list) == 5
    assert len(wound_mask_list) == 5
    assert len(wound_region_list) == 5
    for img in tissue_mask_list:
        assert np.max(img) == 1
        assert np.min(img) == 0
    for img in wound_mask_list:
        assert np.max(img) == 1
        assert np.min(img) == 0


def test_contour_all_and_wound_parameters_all_and_tissue_parameters_all():
    folder_path = example_path("test_mini_movie")
    path_dict = ia.input_info_to_input_paths(folder_path)
    folder_path = path_dict["brightfield_images_path"]
    tiff_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = 1
    thresholded_list = ia.threshold_all(tiff_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, wound_region_list = ia.mask_all(thresholded_list, 1)
    contour_list = ia.contour_all(wound_mask_list)
    area_list, axis_major_length_list, axis_minor_length_list = ia.wound_parameters_all(wound_region_list)
    tissue_parameter_list = ia.tissue_parameters_all(tissue_mask_list, wound_mask_list)
    assert len(tissue_mask_list) == 5
    assert len(contour_list) == 5
    assert len(wound_mask_list) == 5
    assert len(area_list) == 5
    assert len(axis_major_length_list) == 5
    assert len(axis_minor_length_list) == 5
    assert len(tissue_parameter_list) == 5
    assert np.max(area_list) < 512 * 512
    assert np.min(area_list) >= 0
    assert np.max(axis_major_length_list) < 512
    assert np.min(axis_major_length_list) >= 0
    assert np.max(axis_minor_length_list) < 512
    assert np.min(axis_minor_length_list) >= 0
    for kk in range(0, 5):
        assert axis_major_length_list[kk] >= axis_minor_length_list[kk]


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
            wound_name_list, tissue_name_list, contour_name_list, area_path, ax_maj_path, ax_min_path, tissue_path, is_broken_path, is_closed_path, _, _, _, _, _ = ia.run_segment(input_path, output_path, threshold_function_idx)
            for wn in wound_name_list:
                assert wn.is_file()
            for tn in tissue_name_list:
                assert tn.is_file()
            for cn in contour_name_list:
                assert cn.is_file()
            assert tissue_path.is_file()
            assert area_path.is_file()
            assert ax_maj_path.is_file()
            assert ax_min_path.is_file()
            assert is_broken_path.is_file()
            assert is_closed_path.is_file()


def test_run_segment_fl():
    for name in ["test_single", "test_mini_movie"]:
        for kind in ["fluorescent"]:
            print(name, kind)
            folder_path = example_path(name)
            path_dict = ia.input_info_to_input_paths(folder_path)
            input_path = path_dict[kind + "_images_path"]
            input_dict = ia.input_info_to_input_dict(folder_path)
            path_dict = ia.input_info_to_output_paths(folder_path, input_dict)
            output_path = path_dict["segment_" + kind + "_path"]
            threshold_function_idx = 2
            wound_name_list, tissue_name_list, contour_name_list, area_path, ax_maj_path, ax_min_path, tissue_path, is_broken_path, is_closed_path, _, _, _, _, _ = ia.run_segment(input_path, output_path, threshold_function_idx)
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
    wound_name_list, tissue_name_list, contour_name_list, area_path, ax_maj_path, ax_min_path, tissue_path, is_broken_path, is_closed_path, _, _, _, _, _ = ia.run_segment(input_path, output_path, threshold_function_idx)
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


def test_save_all_img_with_contour_and_create_gif_bf():
    for kind in ["brightfield", "fluorescent"]:
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
        thresholded_list = ia.threshold_all(tiff_list, threshold_function_idx)
        tissue_mask_list, wound_mask_list, wound_region_list = ia.mask_all(thresholded_list, 1)
        is_broken_list = ia.check_broken_tissue_all(tissue_mask_list)
        is_closed_list = ia.check_wound_closed_all(tissue_mask_list, wound_region_list)
        contour_list = ia.contour_all(wound_mask_list)
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


def test_run_all_ph1_broken():
    folder_path = example_path("test_ph1_mini_movie_broken")
    time_all, action_all = ia.run_all(folder_path)
    assert len(time_all) == 4
    assert len(action_all) == 4


def test_check_broken_tissue():
    tissue_mask = np.zeros((10, 10))
    tissue_mask[3:7, 3:7] = 1
    is_broken = ia.check_broken_tissue(tissue_mask)
    assert is_broken is False
    tissue_mask[3:5, 3:5] = 0
    is_broken = ia.check_broken_tissue(tissue_mask)
    assert is_broken is True
    tissue_mask = np.zeros((10, 10))
    is_broken = ia.check_broken_tissue(tissue_mask)
    assert is_broken is True


def test_check_broken_tissue_all():
    tissue_mask_list = []
    for kk in range(0, 3):
        tissue_mask = np.zeros((10, 10))
        tissue_mask[3:7, 3:7] = 1
        tissue_mask_list.append(tissue_mask)
    is_broken_list = ia.check_broken_tissue_all(tissue_mask_list)
    for bb in is_broken_list:
        assert bb is False


def test_is_broken_example():
    folder_path = example_path("test_ph1_mini_movie_broken")
    input_dict, input_path_dict, output_path_dict = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = ia.select_threshold_function(input_dict, False, False, True)
    thresholded_list = ia.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, _, _ = ia.mask_all(thresholded_list, threshold_function_idx)
    is_broken_list = ia.check_broken_tissue_all(tissue_mask_list)
    assert is_broken_list[0] is False
    assert is_broken_list[1] is True
    assert is_broken_list[2] is True
    assert is_broken_list[3] is True


def test_is_broken_unbroken_example():
    folder_path = example_path("test_ph1_mini_movie")
    input_dict, input_path_dict, output_path_dict = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = ia.select_threshold_function(input_dict, False, False, True)
    thresholded_list = ia.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, _, _ = ia.mask_all(thresholded_list, threshold_function_idx)
    is_broken_list = ia.check_broken_tissue_all(tissue_mask_list)
    for bb in is_broken_list:
        assert bb is False


def test_check_wound_closed_is_open():
    folder_path = example_path("test_ph1_mini_movie")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = 3
    thresholded_list = ia.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, _, wound_region_list = ia.mask_all(thresholded_list, threshold_function_idx)
    check_closed = ia.check_wound_closed(tissue_mask_list[0], wound_region_list[0])
    assert check_closed is False


def test_get_mean_center():
    array = np.zeros((10, 10))
    array[2:7, 4:6] = 1
    center_0, center_1 = ia.get_mean_center(array)
    assert center_0 == 4
    assert center_1 == 4.5


def test_check_wound_closed_is_closed():
    # check a closed example
    folder_path = example_path("test_mini_movie_closing")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["brightfield_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = 1
    thresholded_list = ia.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, _, wound_region_list = ia.mask_all(thresholded_list, threshold_function_idx)
    check_closed = ia.check_wound_closed(tissue_mask_list[5], wound_region_list[5])
    assert check_closed is True


def test_check_wound_closed_all():
    folder_path = example_path("test_mini_movie_closing")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["brightfield_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = 1
    thresholded_list = ia.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, _, wound_region_list = ia.mask_all(thresholded_list, threshold_function_idx)
    check_closed_list = ia.check_wound_closed_all(tissue_mask_list, wound_region_list)
    for kk in range(0, 3):
        assert check_closed_list[kk] is False
    for kk in range(3, 7):
        assert check_closed_list[kk] is True


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


def test_single_masks_ph1_special_cases():
    file_path = glob_ph1("test_single_fail")[0]
    example_file = io.imread(file_path)
    thresh_img = ia.threshold_array(example_file, 4)
    tissue_mask, wound_mask, wound_region = ia.isolate_masks(thresh_img, 4)
    is_broken = ia.check_broken_tissue(tissue_mask)
    is_closed = ia.check_wound_closed(tissue_mask, wound_region)
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


def test_run_all_ph1_anish():
    folder_path = example_path("test_ph1_movie_mini_Anish")
    time_all, action_all = ia.run_all(folder_path)
    assert len(time_all) == 4
    assert len(action_all) == 4


def test_run_all_ph1_many_examples_anish():
    folder_path = example_path("test_phi_many_examples_Anish")
    time_all, action_all = ia.run_all(folder_path)
    assert len(time_all) == 4
    assert len(action_all) == 4


def test_fill_tissue_mask_reconstruction():
    mask = np.zeros((50, 50))
    mask[14:34, 14:34] = 1
    mask_rect = np.copy(mask)
    mask[20:30, 20:30] = 0
    new_mask = ia.fill_tissue_mask_reconstruction(mask)
    mask_rect[10:38, 10:38] = 1
    assert np.allclose(new_mask, mask_rect)


def test_ix_loop():
    val = 5
    num_pts_contour = 10
    val_new = ia.ix_loop(val, num_pts_contour)
    assert val == val_new
    val = 11
    val_new = ia.ix_loop(val, num_pts_contour)
    assert val_new == 1
    val = -3
    val_new = ia.ix_loop(val, num_pts_contour)
    assert val_new == 7


def test_get_contour_distance_across():
    rad = 50
    disk = morphology.disk(rad, bool)
    array = np.zeros((rad * 5, rad * 5))
    array[rad:rad + disk.shape[0], rad:rad + disk.shape[1]] = disk
    contour = ia.mask_to_contour(array)
    num_pts_contour = contour.shape[0]
    tolerence_check = 0.1
    c_idx = 0
    include_idx = list(range(0, contour.shape[0]))
    distance, ix_opposite = ia.get_contour_distance_across(c_idx, contour, num_pts_contour, include_idx, tolerence_check)
    assert distance < rad * 2.0
    assert distance > rad * 2.0 * 0.9
    assert ix_opposite > num_pts_contour * 0.25
    assert ix_opposite < num_pts_contour * 0.75


def test_get_contour_distance_across_all():
    val1 = 30
    val2 = 35
    array = np.zeros((val2 * 5, val2 * 5))
    array[val2:val2 * 2, val1:val1 * 2] = 1
    contour = ia.mask_to_contour(array)
    regions_all = ia.get_region_props(array)
    region = ia.get_largest_regions(regions_all, 1)[0]
    _, tissue_axis_major_length, tissue_axis_minor_length, centroid_row, centroid_col, _, _, orientation = ia.extract_region_props(region)
    include_idx = ia.include_points_contour(contour, centroid_row, centroid_col, tissue_axis_major_length, tissue_axis_minor_length)
    distance_all, ix_all = ia.get_contour_distance_across_all(contour, include_idx)
    assert np.min(distance_all) < val1 * 1.05
    assert np.max(distance_all[distance_all < math.inf]) < val2 * 1.05
    # sum_all = []
    # for kk in range(0, len(ix_all)):
    #     sum_all.append(kk + ix_all[kk])
    # assert np.min(sum_all) > contour.shape[0] * 0.1


def test_get_contour_width():
    val1 = 100
    val2 = 200
    array = np.zeros((val2 * 5, val2 * 5))
    array[val2:val2 * 2, val1:val1 * 2] = 1
    contour = ia.mask_to_contour(array)
    regions_all = ia.get_region_props(array)
    region = ia.get_largest_regions(regions_all, 1)[0]
    _, tissue_axis_major_length, tissue_axis_minor_length, centroid_row, centroid_col, _, _, orientation = ia.extract_region_props(region)
    width, idx_a, idx_b = ia.get_contour_width(contour, centroid_row, centroid_col, tissue_axis_major_length, tissue_axis_minor_length, orientation)
    assert width < val1
    assert width > val1 * 0.9
    p0a = contour[idx_a, 0]
    p1a = contour[idx_a, 1]
    p0b = contour[idx_b, 0]
    p1b = contour[idx_b, 1]
    dist = ((p0a - p0b)**2.0 + (p1a - p1b)**2.0) ** 0.5
    assert pytest.approx(dist, .1) == width


def test_include_points_contour():
    val1 = 100
    val2 = 200
    array = np.zeros((val2 * 10, val2 * 10))
    array[val2:val2 * 2, val1:val1 * 2] = 1
    contour = ia.mask_to_contour(array)
    regions_all = ia.get_region_props(array)
    region = ia.get_largest_regions(regions_all, 1)[0]
    _, tissue_axis_major_length, tissue_axis_minor_length, centroid_row, centroid_col, _, _, orientation = ia.extract_region_props(region)
    include_idx = ia.include_points_contour(contour, centroid_row, centroid_col, tissue_axis_major_length, tissue_axis_minor_length)
    include_idx = np.asarray(include_idx)
    # import matplotlib.pyplot as plt
    # plt.plot(contour[:, 0], contour[:, 1], 'r-')
    # plt.plot(contour[include_idx, 0], contour[include_idx, 1], 'co')
    for kk in range(0, include_idx.shape[0]):
        di = ia.compute_distance(contour[include_idx[kk], 0], contour[include_idx[kk], 1], centroid_row, centroid_col)
        assert di < 0.25 * (tissue_axis_major_length + tissue_axis_minor_length)


def test_get_local_curvature():
    rad = 50
    disk = morphology.disk(rad, bool)
    array = np.zeros((rad * 4, rad * 4))
    array[rad:rad + disk.shape[0], rad:rad + disk.shape[1]] = disk
    contour = ia.mask_to_contour(array)
    sample_dist = np.min([100, contour.shape[0] * 0.1])
    c_idx = 0
    kappa_1 = ia.get_local_curvature(contour, array, c_idx, sample_dist)
    assert pytest.approx(kappa_1, 0.05) == 1.0 / rad
    rad = 100
    disk = morphology.disk(rad, bool)
    array = np.zeros((rad * 4, rad * 4))
    array[rad:rad + disk.shape[0], rad:rad + disk.shape[1]] = disk
    contour = ia.mask_to_contour(array)
    sample_dist = np.min([100, contour.shape[0] * 0.1])
    c_idx = 0
    kappa_2 = ia.get_local_curvature(contour, array, c_idx, sample_dist)
    assert pytest.approx(kappa_2, 0.05) == 1.0 / rad
    assert kappa_1 > kappa_2
    array = np.zeros((30, 1000))
    array[10:20, 100:900] = 1
    contour = ia.mask_to_contour(array)
    sample_dist = np.min([100, contour.shape[0] * 0.1])
    c_idx = 500
    kappa_1 = ia.get_local_curvature(contour, array, c_idx, sample_dist)
    assert math.isinf(kappa_1)


def test_insert_borders():
    mask = np.ones((50, 50))
    border = 10
    mask = ia.insert_borders(mask, border)
    assert np.sum(mask) == 30 * 30


def test_line_param():
    array = np.zeros((50, 50))
    array[5:40, 10:15] = 1
    regions_all = ia.get_region_props(array)
    region = ia.get_largest_regions(regions_all, 1)[0]
    _, axis_major_length, _, centroid_row, centroid_col, _, _, orientation = ia.extract_region_props(region)
    line_a, line_b, line_c = ia.line_param(centroid_row, centroid_col, orientation)
    assert pytest.approx(line_a, 0.01) == 0.0
    assert pytest.approx(line_b, 0.01) == 1.0
    array = np.zeros((50, 50))
    array[10:15, 5:40] = 1
    regions_all = ia.get_region_props(array)
    region = ia.get_largest_regions(regions_all, 1)[0]
    _, axis_major_length, _, centroid_row, centroid_col, _, _, orientation = ia.extract_region_props(region)
    line_a, line_b, line_c = ia.line_param(centroid_row, centroid_col, orientation)
    assert np.abs(line_a) > 1.0 * 10 ** 10.0
    assert pytest.approx(line_b, 0.01) == 1.0
    array = np.zeros((50, 50))
    array[10:15, 10:15] = 1
    array[14:20, 14:20] = 1
    regions_all = ia.get_region_props(array)
    region = ia.get_largest_regions(regions_all, 1)[0]
    _, axis_major_length, _, centroid_row, centroid_col, _, _, orientation = ia.extract_region_props(region)
    line_a, line_b, line_c = ia.line_param(centroid_row, centroid_col, orientation)
    assert pytest.approx(line_a, 0.01) == pytest.approx(line_b, 0.01)
    # import matplotlib.pyplot as plt
    # x = np.linspace(0, array.shape[0])
    # y = -1.0 * line_a / line_b * x - line_c / line_b
    # plt.imshow(array)
    # plt.plot(x, y, 'r')
    # aa = 44


def test_dist_to_line():
    array = np.zeros((50, 50))
    array[5:40, 10:15] = 1
    regions_all = ia.get_region_props(array)
    region = ia.get_largest_regions(regions_all, 1)[0]
    _, axis_major_length, _, centroid_row, centroid_col, _, _, orientation = ia.extract_region_props(region)
    line_a, line_b, line_c = ia.line_param(centroid_row, centroid_col, orientation)
    pt_0 = 0.0
    pt_1 = 22.0
    line_dist = ia.dist_to_line(line_a, line_b, line_c, pt_0, pt_1)
    assert pytest.approx(line_dist, 0.01) == 0.0
    array = np.zeros((50, 50))
    array[10:15, 10:15] = 1
    array[14:20, 14:20] = 1
    regions_all = ia.get_region_props(array)
    region = ia.get_largest_regions(regions_all, 1)[0]
    _, axis_major_length, _, centroid_row, centroid_col, _, _, orientation = ia.extract_region_props(region)
    line_a, line_b, line_c = ia.line_param(centroid_row, centroid_col, orientation)
    pt_0 = 16.08088022903976
    pt_1 = 16.08088022903976
    line_dist = ia.dist_to_line(line_a, line_b, line_c, pt_0, pt_1)
    assert pytest.approx(line_dist, 0.01) == 2.0
    pt_0 = 13.25245310429357
    pt_1 = 13.25245310429357
    line_dist = ia.dist_to_line(line_a, line_b, line_c, pt_0, pt_1)
    assert pytest.approx(line_dist, 0.01) == 2.0


def test_clip_contour():
    rad = 50
    disk = morphology.disk(rad, bool)
    array = np.zeros((rad * 4, rad * 4))
    array[rad:rad + disk.shape[0], rad:rad + disk.shape[1]] = disk
    contour = ia.mask_to_contour(array)
    regions_all = ia.get_region_props(array)
    region = ia.get_largest_regions(regions_all, 1)[0]
    _, axis_major_length, axis_minor_length, centroid_row, centroid_col, _, _, orientation = ia.extract_region_props(region)
    contour_clipped = ia.clip_contour(contour, centroid_row, centroid_col, orientation, axis_major_length * 2.0, axis_minor_length * 2.0)
    assert np.allclose(contour, contour_clipped)
    array = np.zeros((50, 50))
    array[10:40, 20:25] = 1
    contour = ia.mask_to_contour(array)
    regions_all = ia.get_region_props(array)
    region = ia.get_largest_regions(regions_all, 1)[0]
    _, axis_major_length, axis_minor_length, centroid_row, centroid_col, _, _, orientation = ia.extract_region_props(region)
    contour_clipped = ia.clip_contour(contour, centroid_row, centroid_col, orientation, axis_major_length, axis_minor_length)
    # import matplotlib.pyplot as plt
    # plt.plot(contour[:,0], contour[:,1])
    # plt.plot(contour_clipped[:,0], contour_clipped[:,1],'r')
    # plt.axis('equal')
    assert np.allclose(contour_clipped, contour) is False


def test_move_point():
    pt_0 = -10
    pt_1 = 10
    line_a = 1
    line_b = 1
    line_c = 0
    cutoff = 5
    pt_0_new, pt_1_new = ia.move_point(pt_0, pt_1, line_a, line_b, line_c, cutoff)
    assert pt_0_new == -10
    assert pt_1_new == 10
    pt_0 = -10
    pt_1 = 10
    line_a = -1
    line_b = 1
    line_c = 0
    cutoff = 5 * np.sqrt(2)
    pt_0_new, pt_1_new = ia.move_point(pt_0, pt_1, line_a, line_b, line_c, cutoff)
    assert pytest.approx(pt_0_new, .01) == -5
    assert pytest.approx(pt_1_new, .01) == 5
    pt_0 = 10
    pt_1 = -10
    line_a = -1
    line_b = 1
    line_c = 0
    cutoff = 5 * np.sqrt(2)
    pt_0_new, pt_1_new = ia.move_point(pt_0, pt_1, line_a, line_b, line_c, cutoff)
    assert pytest.approx(pt_0_new, .01) == 5
    assert pytest.approx(pt_1_new, .01) == -5
    pt_0 = 10
    pt_1 = 10
    line_a = 1
    line_b = 1
    line_c = 0
    cutoff = 5 * np.sqrt(2)
    pt_0_new, pt_1_new = ia.move_point(pt_0, pt_1, line_a, line_b, line_c, cutoff)
    assert pytest.approx(pt_0_new, .01) == 5
    assert pytest.approx(pt_1_new, .01) == 5
    pt_0 = -10
    pt_1 = -10
    line_a = 1
    line_b = 1
    line_c = 0
    cutoff = 5 * np.sqrt(2)
    pt_0_new, pt_1_new = ia.move_point(pt_0, pt_1, line_a, line_b, line_c, cutoff)
    assert pytest.approx(pt_0_new, .01) == -5
    assert pytest.approx(pt_1_new, .01) == -5
    pt_0 = 10
    pt_1 = 10
    line_a = 1
    line_b = 0
    line_c = 0
    cutoff = 5
    pt_0_new, pt_1_new = ia.move_point(pt_0, pt_1, line_a, line_b, line_c, cutoff)
    assert pytest.approx(pt_0_new, .01) == 10
    assert pytest.approx(pt_1_new, .01) == 5
    pt_0 = 10
    pt_1 = 10
    line_a = 0
    line_b = 1
    line_c = 0
    cutoff = 5
    pt_0_new, pt_1_new = ia.move_point(pt_0, pt_1, line_a, line_b, line_c, cutoff)
    assert pytest.approx(pt_0_new, .01) == 5
    assert pytest.approx(pt_1_new, .01) == 10


def test_resample_contour():
    array = np.zeros((50, 50))
    array[10:40, 20:25] = 1
    contour = ia.mask_to_contour(array)
    regions_all = ia.get_region_props(array)
    region = ia.get_largest_regions(regions_all, 1)[0]
    _, axis_major_length, axis_minor_length, centroid_row, centroid_col, _, _, orientation = ia.extract_region_props(region)
    contour_clipped = ia.clip_contour(contour, centroid_row, centroid_col, orientation, axis_major_length, axis_minor_length)
    resampled_contour = ia.resample_contour(contour_clipped)
    assert resampled_contour.shape[0] == contour_clipped.shape[0]
    array = np.zeros((1000, 1000))
    array[200:800, 200:250] = 1
    contour = ia.mask_to_contour(array)
    regions_all = ia.get_region_props(array)
    region = ia.get_largest_regions(regions_all, 1)[0]
    _, axis_major_length, axis_minor_length, centroid_row, centroid_col, _, _, orientation = ia.extract_region_props(region)
    contour_clipped = ia.clip_contour(contour, centroid_row, centroid_col, orientation, axis_major_length, axis_minor_length)
    resampled_contour = ia.resample_contour(contour_clipped)
    assert resampled_contour.shape[0] < contour_clipped.shape[0]
    # make test more robust!
    # import matplotlib.pyplot as plt
    # plt.plot(contour[:,0], contour[:,1])
    # plt.plot(contour_clipped[:,0], contour_clipped[:,1],'r-')
    # plt.plot(resampled_contour[:,0], resampled_contour[:,1],'g.')
    # plt.axis('equal')


def test_tissue_parameters():
    folder_path = example_path("test_ph1_mini_movie")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    img_list = [img_list[0]]
    threshold_function_idx = 3
    thresholded_list = ia.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, _ = ia.mask_all(thresholded_list, threshold_function_idx)
    width, area, kappa_1, kappa_2, pt1_0, pt1_1, pt2_0, pt2_1, tissue_contour = ia.tissue_parameters(tissue_mask_list[0], wound_mask_list[0])
    # tissue_robust = ia.make_tissue_mask_robust(tissue_mask_list[0], wound_mask_list[0])
    # regions_all = ia.get_region_props(tissue_robust)
    # region = ia.get_largest_regions(regions_all, 1)[0]
    # _, axis_major_length, axis_minor_length, centroid_row, centroid_col, _, _, orientation = ia.extract_region_props(region)
    # contour_clipped = ia.clip_contour(tissue_contour, centroid_row, centroid_col, orientation, axis_major_length, axis_minor_length)
    # import matplotlib.pyplot as plt
    # plt.imshow(img_list[0])
    # plt.plot(tissue_contour[:,1], tissue_contour[:,0],'r-o')
    # # plt.plot(contour_clipped[:,1], contour_clipped[:,0],'c-.')
    # plt.plot(pt1_1, pt1_0,'bo')
    # plt.plot(pt2_1, pt2_0, 'go')
    assert width > 0
    assert area > 0
    assert kappa_1 < 1
    assert kappa_2 < 1
    assert pt1_0 > 0 and pt1_0 < img_list[0].shape[1]
    assert pt2_0 > 0 and pt2_0 < img_list[0].shape[1]
    assert pt1_1 > 0 and pt1_1 < img_list[0].shape[0]
    assert pt2_1 > 0 and pt2_1 < img_list[0].shape[0]
    assert tissue_contour.shape[0] > 0
    # import matplotlib.pyplot as plt
    # plt.imshow(img_list[0])
    # plt.plot(tissue_contour[:,1], tissue_contour[:,0],'r-o')
    # plt.plot(pt1_1, pt1_0,'bo')
    # plt.plot(pt2_1, pt2_0, 'go')


def test_get_penalized():
    contour = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]])
    contour_clipped = np.asarray([[1, 2], [3, 7], [8, 6], [9, 10]])
    penalized_contour = ia.get_penalized(contour, contour_clipped)
    assert penalized_contour[0, 0] == 1
    assert penalized_contour[0, 1] == 2
    for kk in range(1, 4):
        assert penalized_contour[kk, 0] == math.inf
        assert penalized_contour[kk, 1] == math.inf


def test_tissue_parameters_anish():
    folder_path = example_path("test_ph1_movie_mini_Anish")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    img_list = [img_list[0]]
    threshold_function_idx = 4
    thresholded_list = ia.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, _ = ia.mask_all(thresholded_list, threshold_function_idx)
    width, area, kappa_1, kappa_2, pt1_0, pt1_1, pt2_0, pt2_1, tissue_contour = ia.tissue_parameters(tissue_mask_list[0], wound_mask_list[0])
    # tissue_robust = ia.make_tissue_mask_robust(tissue_mask_list[0], wound_mask_list[0])
    # regions_all = ia.get_region_props(tissue_robust)
    # region = ia.get_largest_regions(regions_all, 1)[0]
    # _, axis_major_length, axis_minor_length, centroid_row, centroid_col, _, _, orientation = ia.extract_region_props(region)
    # contour_clipped = ia.clip_contour(tissue_contour, centroid_row, centroid_col, orientation, axis_major_length, axis_minor_length)
    # contour_clipped_2 = ia.clip_contour(contour_clipped, centroid_row, centroid_col, orientation - np.pi / 2.0, axis_major_length, axis_minor_length)
    # import matplotlib.pyplot as plt
    # plt.imshow(img_list[0])
    # plt.plot(tissue_contour[:,1], tissue_contour[:,0],'r-o')
    # plt.plot(contour_clipped_2[:,1], contour_clipped_2[:,0],'c-.')
    # plt.plot(pt1_1, pt1_0, 'bo')
    # plt.plot(pt2_1, pt2_0, 'go')
    assert width > 0
    assert area > 0
    assert kappa_1 < 1
    assert kappa_2 < 1
    assert pt1_0 > 0 and pt1_0 < img_list[0].shape[1]
    assert pt2_0 > 0 and pt2_0 < img_list[0].shape[1]
    assert pt1_1 > 0 and pt1_1 < img_list[0].shape[0]
    assert pt2_1 > 0 and pt2_1 < img_list[0].shape[0]
    assert tissue_contour.shape[0] > 100


def test_make_tissue_mask_robust():
    folder_path = example_path("test_ph1_movie_mini_Anish")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    img_list = [img_list[0]]
    threshold_function_idx = 4
    thresholded_list = ia.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, _ = ia.mask_all(thresholded_list, threshold_function_idx)
    tissue_mask = tissue_mask_list[0]
    wound_mask = wound_mask_list[0]
    tissue_mask_robust = ia.make_tissue_mask_robust(tissue_mask, wound_mask)
    tissue_contour = ia.mask_to_contour(tissue_mask_robust)
    assert tissue_contour.shape[0] > 100


def test_make_tissue_mask_robust_brightfield():
    folder_path = example_path("test_mini_movie")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["brightfield_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    img_list = [img_list[0]]
    threshold_function_idx = 4
    thresholded_list = ia.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, _ = ia.mask_all(thresholded_list, threshold_function_idx)
    tissue_mask = tissue_mask_list[0]
    wound_mask = wound_mask_list[0]
    tissue_mask_robust = ia.make_tissue_mask_robust(tissue_mask, wound_mask)
    tissue_contour = ia.mask_to_contour(tissue_mask_robust)
    assert tissue_contour.shape[0] > 100


def test_get_tracking_param_dicts():
    feature_params, lk_params = ia.get_tracking_param_dicts()
    assert feature_params["maxCorners"] == 1000
    assert feature_params["qualityLevel"] == 0.1
    assert feature_params["minDistance"] == 7
    assert feature_params["blockSize"] == 7
    assert lk_params["winSize"][0] == 50
    assert lk_params["winSize"][1] == 50
    assert lk_params["maxLevel"] == 10
    assert lk_params["criteria"][1] == 10
    assert lk_params["criteria"][2] == 0.03


def test_get_unique():
    numbers = [1, 1, 2, 3, 3, 3, 3, 4, 5]
    list_unique = ia.get_unique(numbers)
    assert len(list_unique) == 5
    assert 1 in list_unique
    assert 2 in list_unique
    assert 3 in list_unique
    assert 4 in list_unique
    assert 5 in list_unique
    numbers = [4, 4, 4, 3, 2, 2, 5, 1, 1, 1, 1]
    list_unique = ia.get_unique(numbers)
    assert len(list_unique) == 5
    assert 1 in list_unique
    assert 2 in list_unique
    assert 3 in list_unique
    assert 4 in list_unique
    assert 5 in list_unique


def test_get_order_track():
    len_img_list = 3
    is_forward = True
    order_list = ia.get_order_track(len_img_list, is_forward)
    for kk in range(0, len_img_list):
        assert order_list[kk] is kk
    is_forward = False
    order_list = ia.get_order_track(len_img_list, is_forward)
    for kk in range(0, len_img_list):
        assert order_list[kk] is len_img_list - kk - 1


def test_bool_to_uint8():
    arr_bool = np.random.random((10, 10)) > 0.5
    arr_uint8 = ia.bool_to_uint8(arr_bool)
    assert np.max(arr_uint8) == 1
    assert np.min(arr_uint8) == 0
    assert arr_uint8.dtype == np.dtype("uint8")


def test_mask_to_track_points_and_track_one_step_and_track_all_steps():
    folder_path = example_path("test_phi_movie_mini_Anish_tracking")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list_orig = ia.read_all_tiff(folder_path)
    img_list = [img_list_orig[0]]
    threshold_function_idx = 4
    thresholded_list = ia.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, _, _ = ia.mask_all(thresholded_list, threshold_function_idx)
    tissue_mask = tissue_mask_list[0]
    # wound_mask = wound_mask_list[0]
    # wound_contour = ia.mask_to_contour(wound_mask)
    img_uint8 = ia.uint16_to_uint8(img_list[0])
    feature_params, lk_params = ia.get_tracking_param_dicts()
    track_points_0 = ia.mask_to_track_points(img_uint8, tissue_mask, feature_params)
    assert track_points_0.shape[2] == 2
    mask_ix = np.nonzero(tissue_mask > 0)
    mask_ix_0_max = np.max(mask_ix[0])
    mask_ix_0_min = np.min(mask_ix[0])
    mask_ix_1_max = np.max(mask_ix[1])
    mask_ix_1_min = np.min(mask_ix[1])
    assert np.min(track_points_0[:, 0, 0]) > mask_ix_1_min  # note flipped indicies
    assert np.max(track_points_0[:, 0, 0]) < mask_ix_1_max  # note flipped indicies
    assert np.min(track_points_0[:, 0, 1]) > mask_ix_0_min  # note flipped indicies
    assert np.max(track_points_0[:, 0, 1]) < mask_ix_0_max  # note flipped indicies
    img_uint8_0 = ia.uint16_to_uint8(img_list_orig[0])
    img_uint8_1 = ia.uint16_to_uint8(img_list_orig[1])
    track_points_1 = ia.track_one_step(img_uint8_0, img_uint8_1, track_points_0, lk_params)
    assert track_points_0.shape == track_points_1.shape
    diff_0 = np.abs(track_points_0[:, 0, 0] - track_points_1[:, 0, 0])
    diff_1 = np.abs(track_points_0[:, 0, 1] - track_points_1[:, 0, 1])
    window_0 = lk_params["winSize"][0]
    window_1 = lk_params["winSize"][1]
    assert np.max(diff_0) < window_0
    assert np.max(diff_1) < window_1
    img_list_uint8 = ia.uint16_to_uint8_all(img_list_orig)
    order_list = ia.get_order_track(len(img_list_uint8), True)
    tracker_x, tracker_y = ia.track_all_steps(img_list_uint8, tissue_mask, order_list)
    assert tracker_x.shape[1] == len(img_list_uint8)
    assert tracker_y.shape[1] == len(img_list_uint8)
    assert tracker_x.shape[0] == track_points_0.shape[0]
    assert tracker_y.shape[0] == track_points_0.shape[0]
    assert np.allclose(tracker_x[:, 0], track_points_0[:, 0, 0])
    assert np.allclose(tracker_y[:, 0], track_points_0[:, 0, 1])


def test_wound_mask_from_points():
    folder_path = example_path("test_phi_movie_mini_Anish_tracking")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list_orig = ia.read_all_tiff(folder_path)
    img_list = [img_list_orig[0], img_list_orig[1], img_list_orig[2]]
    threshold_function_idx = 4
    thresholded_list = ia.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, _ = ia.mask_all(thresholded_list, threshold_function_idx)
    tissue_mask = tissue_mask_list[0]
    wound_mask = wound_mask_list[0]
    wound_contour = ia.mask_to_contour(wound_mask)
    frame_0_mask = img_list[0]
    img_list_uint8 = ia.uint16_to_uint8_all(img_list)
    order_list = ia.get_order_track(len(img_list_uint8), True)
    tracker_x, tracker_y = ia.track_all_steps(img_list_uint8, tissue_mask, order_list)
    alpha_assigned = False
    mask_wound_initial, mask_wound_final = ia.wound_mask_from_points(frame_0_mask, tracker_x, tracker_y, wound_contour, alpha_assigned)
    assert mask_wound_initial.shape == mask_wound_final.shape
    assert mask_wound_initial.shape == frame_0_mask.shape
    assert np.max(mask_wound_initial) == 1
    assert np.min(mask_wound_initial) == 0
    assert np.max(mask_wound_final) == 1
    assert np.min(mask_wound_final) == 0
    alpha_assigned = True
    assigned_alpha = 0.015
    mask_wound_initial, mask_wound_final = ia.wound_mask_from_points(frame_0_mask, tracker_x, tracker_y, wound_contour, alpha_assigned, assigned_alpha)
    assert mask_wound_initial.shape == mask_wound_final.shape
    assert mask_wound_initial.shape == frame_0_mask.shape
    assert np.max(mask_wound_initial) == 1
    assert np.min(mask_wound_initial) == 0
    assert np.max(mask_wound_final) == 1
    assert np.min(mask_wound_final) == 0


def test_perform_tracking():
    folder_path = example_path("test_phi_movie_mini_Anish_tracking")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = 4
    thresholded_list = ia.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, _ = ia.mask_all(thresholded_list, threshold_function_idx)
    frame_0_mask = tissue_mask_list[0]
    include_reverse = True
    wound_mask = wound_mask_list[0]
    wound_contour = ia.mask_to_contour(wound_mask)
    tracker_x, tracker_y, tracker_x_reverse, tracker_y_reverse = ia.perform_tracking(frame_0_mask, img_list, include_reverse, wound_contour)
    include_reverse = False
    tracker_x_forward, tracker_y_forward, tracker_x_reverse_forward, tracker_y_reverse_forward = ia.perform_tracking(frame_0_mask, img_list, include_reverse, wound_contour)
    assert tracker_x.shape[1] == len(img_list)
    assert tracker_y.shape[1] == len(img_list)
    assert tracker_x_reverse.shape[1] == len(img_list)
    assert tracker_y_reverse.shape[1] == len(img_list)
    assert tracker_x_forward.shape[1] == len(img_list)
    assert tracker_y_forward.shape[1] == len(img_list)
    assert tracker_x_reverse_forward is None
    assert tracker_y_reverse_forward is None
