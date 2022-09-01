import numpy as np
from pathlib import Path
from scipy import ndimage
from skimage import io
from skimage import morphology
from woundcompute import image_analysis as ia


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


def test_threshold_gfp_v1():
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    file_path = data_path.joinpath("test_gfp.TIF")
    example_file = io.imread(file_path)
    thresh_img = ia.threshold_gfp_v1(example_file)
    assert np.max(thresh_img) == 1
    assert np.min(thresh_img) == 0
    assert thresh_img.shape[0] == example_file.shape[0]
    assert thresh_img.shape[1] == example_file.shape[1]


def test_threshold_brightfield_v1():
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    file_path = data_path.joinpath("test_brightfield.TIF")
    example_file = io.imread(file_path)
    thresh_img = ia.threshold_brightfield_v1(example_file)
    assert np.max(thresh_img) == 1
    assert np.min(thresh_img) == 0
    assert thresh_img.shape[0] == example_file.shape[0]
    assert thresh_img.shape[1] == example_file.shape[1]


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
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    file_path = data_path.joinpath("test_gfp.TIF")
    example_file = io.imread(file_path)
    thresh_img = ia.threshold_gfp_v1(example_file)
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
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    file_path = data_path.joinpath("test_brightfield.TIF")
    example_file = io.imread(file_path)
    thresh_img = ia.threshold_brightfield_v1(example_file)
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
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    file_path = data_path.joinpath("test_brightfield.TIF")
    known = io.imread(file_path)
    found = ia.read_tiff(file_path)
    assert np.all(known == found)


def test_show_and_save_image():
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    file_path = data_path.joinpath("test_brightfield.TIF")
    file = ia.read_tiff(file_path)
    save_path = data_path.joinpath("test_brightfield_save_no_title.png")
    ia.show_and_save_image(file, save_path)
    assert save_path.is_file()
    save_path_title = data_path.joinpath("test_brightfield_save_title.png")
    title = 'test brightfield title'
    ia.show_and_save_image(file, save_path_title, title)
    assert save_path_title.is_file()


def test_show_and_save_image_mask():
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    file_path = data_path.joinpath("test_brightfield.TIF")
    file = ia.read_tiff(file_path)
    file_thresh = ia.threshold_brightfield_v1(file)
    tissue_mask, wound_mask, wound_region = ia.isolate_masks(file_thresh)
    save_path = data_path.joinpath("test_brightfield_tissue_mask.png")
    ia.show_and_save_image(tissue_mask, save_path)
    assert save_path.is_file()
    save_path = data_path.joinpath("test_brightfield_wound_mask.png")
    ia.show_and_save_image(wound_mask, save_path)
    assert save_path.is_file()
    file_path = data_path.joinpath("test_gfp.TIF")
    file = ia.read_tiff(file_path)
    file_thresh = ia.threshold_gfp_v1(file)
    tissue_mask, wound_mask, wound_region = ia.isolate_masks(file_thresh)
    save_path = data_path.joinpath("test_gfp_tissue_mask.png")
    ia.show_and_save_image(tissue_mask, save_path)
    assert save_path.is_file()
    save_path = data_path.joinpath("test_gfp_wound_mask.png")
    ia.show_and_save_image(wound_mask, save_path)
    assert save_path.is_file()


def test_show_and_save_contour():
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    file_path = data_path.joinpath("test_brightfield.TIF")
    file = ia.read_tiff(file_path)
    file_thresh = ia.threshold_brightfield_v1(file)
    wound_mask = ia.isolate_masks(file_thresh)[1]
    contour = ia.mask_to_contour(wound_mask)
    save_path = data_path.joinpath("test_brightfield_wound_contour.png")
    ia.show_and_save_contour(file, contour, save_path)
    assert save_path.is_file()


def test_save_numpy():
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    file_path = data_path.joinpath("test_brightfield.TIF")
    file = ia.read_tiff(file_path)
    save_path = data_path.joinpath("test_brightfield_save_no_title.npy")
    ia.save_numpy(file, save_path)
    assert save_path.is_file()
    file_thresh = ia.threshold_brightfield_v1(file)
    tissue_mask, wound_mask, wound_region = ia.isolate_masks(file_thresh)
    save_path = data_path.joinpath("test_brightfield_tissue_mask.npy")
    ia.save_numpy(tissue_mask, save_path)
    assert save_path.is_file()
    contour = ia.mask_to_contour(wound_mask)
    save_path = data_path.joinpath("test_brightfield_tissue_contour.npy")
    ia.save_numpy(contour, save_path)
    assert save_path.is_file()


def test_save_yaml():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    region_props = ia.get_region_props(disk_1)
    region = region_props[0]
    area, axis_major_length, axis_minor_length, centroid_row, centroid_col, coords = ia.extract_region_props(region)
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    file_path = data_path.joinpath("test_save.yaml")
    ia.save_yaml(area, axis_major_length, axis_minor_length, centroid_row, centroid_col, file_path)
    assert file_path.is_file()


def test_analyze_image():
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    img_path = data_path.joinpath("test_brightfield.TIF")
    is_brightfield = True
    tissue_mask_path = data_path.joinpath("test_brightfield_tissue_mask.npy")
    wound_mask_path = data_path.joinpath("test_brightfield_wound_mask.npy")
    contour_path = data_path.joinpath("test_brightfield_contour.npy")
    yaml_path = data_path.joinpath("test_brightfield_values.yaml")
    vis_path = data_path.joinpath("test_brightfield_visualize.png")
    ia.analyze_image(img_path, is_brightfield, tissue_mask_path, wound_mask_path, contour_path, yaml_path, vis_path)
    assert tissue_mask_path.is_file()
    assert wound_mask_path.is_file()
    assert contour_path.is_file()
    assert yaml_path.is_file()
    assert vis_path.is_file()
