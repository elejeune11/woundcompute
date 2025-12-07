import glob
import math
import numpy as np
from pathlib import Path
import pytest
from scipy.spatial import distance
from skimage import io
from skimage import morphology
from woundcompute import image_analysis as ia
from woundcompute import compute_values as com
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


def test_compute_distance():
    x1 = 0
    x2 = 10
    y1 = 0
    y2 = 0
    dist = com.compute_distance(x1, x2, y1, y2)
    assert np.isclose(dist, 10)


def test_compute_unit_vector():
    x1 = 0
    x2 = 10
    y1 = 0
    y2 = 0
    vec = com.compute_unit_vector(x1, x2, y1, y2)
    assert np.allclose(vec, np.asarray([1, 0]))


def test_compute_distance_multi_point():
    coords_1 = np.random.random((10, 2))
    coords_2 = np.random.random((5, 2))
    pt1_0_orig, pt1_1_orig, pt2_0_orig, pt2_1_orig = com.compute_distance_multi_point(coords_1, coords_2)
    arr = distance.cdist(coords_1, coords_2, 'euclidean')
    min_known = np.min(arr)
    x1 = pt1_0_orig
    x2 = pt2_0_orig
    y1 = pt1_1_orig
    y2 = pt2_1_orig
    min_found = ((x1 - x2) ** 2.0 + (y1 - y2) ** 2.0) ** 0.5
    assert np.isclose(min_known, min_found)
    coords_1 = np.ones((2895, 2))
    coords_2 = np.zeros((5567, 2))
    pt1_0_orig, pt1_1_orig, pt2_0_orig, pt2_1_orig = com.compute_distance_multi_point(coords_1, coords_2)
    arr = distance.cdist(coords_1, coords_2, 'euclidean')
    min_known = np.min(arr)
    x1 = pt1_0_orig
    x2 = pt2_0_orig
    y1 = pt1_1_orig
    y2 = pt2_1_orig
    min_found = ((x1 - x2) ** 2.0 + (y1 - y2) ** 2.0) ** 0.5
    assert np.isclose(min_known, min_found)


def test_box_to_unit_vec():
    box = np.asarray([[0, 0], [0, 10], [5, 10], [5, 0]])
    vec = com.box_to_unit_vec(box)
    assert np.allclose(vec, np.asarray([0, 1]), atol=.1) or np.allclose(vec, np.asarray([0, -1]), atol=.1)
    box = np.asarray([[0, 0], [0, 5], [10, 5], [10, 0]])
    vec = com.box_to_unit_vec(box)
    assert np.allclose(vec, np.asarray([1, 0])) or np.allclose(vec, np.asarray([-1, 0]))


def test_box_to_center_points():
    box = np.asarray([[0, 0], [0, 10], [5, 10], [5, 0]])
    center_row, center_col = com.box_to_center_points(box)
    assert np.isclose(center_row, 2.5)
    assert np.isclose(center_col, 5.0)


def test_insert_borders():
    mask = np.ones((50, 50))
    border = 10
    mask = com.insert_borders(mask, border)
    assert np.sum(mask) == 30 * 30


def test_ix_loop():
    val = 5
    num_pts_contour = 10
    val_new = com.ix_loop(val, num_pts_contour)
    assert val == val_new
    val = 11
    val_new = com.ix_loop(val, num_pts_contour)
    assert val_new == 1
    val = -3
    val_new = com.ix_loop(val, num_pts_contour)
    assert val_new == 7


def test_get_local_curvature():
    rad = 50
    disk = morphology.disk(rad, bool)
    array = np.zeros((rad * 4, rad * 4))
    array[rad:rad + disk.shape[0], rad:rad + disk.shape[1]] = disk
    contour = seg.mask_to_contour(array)
    sample_dist = np.min([100, contour.shape[0] * 0.1])
    c_idx = 0
    kappa_1 = com.get_local_curvature(contour, array, c_idx, sample_dist)
    assert pytest.approx(kappa_1, 0.05) == 1.0 / rad
    rad = 100
    disk = morphology.disk(rad, bool)
    array = np.zeros((rad * 4, rad * 4))
    array[rad:rad + disk.shape[0], rad:rad + disk.shape[1]] = disk
    contour = seg.mask_to_contour(array)
    sample_dist = np.min([100, contour.shape[0] * 0.1])
    c_idx = 0
    kappa_2 = com.get_local_curvature(contour, array, c_idx, sample_dist)
    assert pytest.approx(kappa_2, 0.05) == 1.0 / rad
    assert kappa_1 > kappa_2
    array = np.zeros((30, 1000))
    array[10:20, 100:900] = 1
    contour = seg.mask_to_contour(array)
    sample_dist = np.min([100, contour.shape[0] * 0.1])
    c_idx = 500
    kappa_1 = com.get_local_curvature(contour, array, c_idx, sample_dist)
    assert math.isinf(kappa_1)


def test_sort_points_counterclockwise():
    points = np.array([[0,0],[1,1],[0,1],[1,0]])
    known = np.array([[0,0],[1,0],[1,1],[0,1]])
    found = com.sort_points_counterclockwise(points)
    assert np.allclose(known,found)
    known = np.array([[0,1],[1,1],[1,0],[0,0]])
    found = com.sort_points_counterclockwise(points,clockwise=True)
    assert np.allclose(known,found)


def test_mask_to_box():
    mask = np.zeros((100, 100))
    mask[25:75, 45:55] = 1
    box = com.mask_to_box(mask)
    assert box.shape == (4, 2)
    assert np.isclose(np.min(box[:, 0]), 25, atol=3)
    assert np.isclose(np.max(box[:, 0]), 74, atol=3)
    assert np.isclose(np.min(box[:, 1]), 45, atol=3)
    assert np.isclose(np.max(box[:, 1]), 54, atol=3)
    mask = np.zeros((100, 100))
    mask[25:75, 45:55] = 1
    box = com.mask_to_box(mask, 1)
    assert box.shape == (4, 2)
    assert np.isclose(np.min(box[:, 0]), 25, atol=5)
    assert np.isclose(np.max(box[:, 0]), 74, atol=5)
    assert np.isclose(np.min(box[:, 1]), 45, atol=5)
    assert np.isclose(np.max(box[:, 1]), 54, atol=5)


def test_axis_from_mask_artifical():
    # create an artificial mask
    mask = np.zeros((100, 100))
    mask[25:75, 45:55] = 1
    center_row, center_col, vec = com.axis_from_mask(mask)
    assert np.allclose(vec, np.asarray([1, 0])) or np.allclose(vec, np.asarray([-1, 0]))
    assert np.isclose(center_row, (25 + 74) / 2.0, atol=2)
    assert np.isclose(center_col, (46 + 53) / 2.0, atol=2)
    mask = np.zeros((100, 100))
    mask[45:55, 25:75] = 1
    center_row, center_col, vec = com.axis_from_mask(mask)
    assert np.allclose(vec, np.asarray([0, 1])) or np.allclose(vec, np.asarray([0, -1]))
    assert np.isclose(center_col, (25 + 74) / 2.0, atol=2)
    assert np.isclose(center_row, (46 + 53) / 2.0, atol=2)


# def test_axis_from_mask_real():
#     file_path = tissue_mask_path("real_example_super_short")
#     mask = ia.read_txt_as_mask(file_path)
#     center_row, center_col, vec = ia.axis_from_mask(mask)
#     assert np.isclose(center_row, mask.shape[0] / 2.0, atol=10)
#     assert np.isclose(center_col, mask.shape[0] / 2.0, atol=10)
#     assert np.allclose(vec, np.asarray([0, 1]), atol=.1) or np.allclose(vec, np.asarray([0, -1]), atol=.1)
#     # rotated example
#     mask = np.zeros((100, 100))
#     for kk in range(10, 50):
#         mask[kk, kk + 20:kk + 30] = 1
#     center_row, center_col, vec = ia.axis_from_mask(mask)
#     assert np.isclose(center_row, (10 + 50) / 2.0, atol=4)
#     assert np.isclose(center_col, (30 + 80) / 2.0, atol=4)
#     assert np.allclose(vec, np.asarray([np.sqrt(2) / 2.0, np.sqrt(2) / 2.0]))

def test_rot_vec_to_rot_mat_and_angle():
    vec = [1, 0]
    (rot_mat, ang) = com.rot_vec_to_rot_mat_and_angle(vec)
    assert np.isclose(ang, np.pi / 2.0)
    assert np.allclose(rot_mat, np.asarray([[0, -1], [1, 0]]))
    vec = [0, 1]
    (rot_mat, ang) = com.rot_vec_to_rot_mat_and_angle(vec)
    assert np.isclose(ang, 0)
    assert np.allclose(rot_mat, np.asarray([[1, 0], [0, 1]]))
    vec = [np.sqrt(2) / 2.0, np.sqrt(2) / 2.0]
    (rot_mat, ang) = com.rot_vec_to_rot_mat_and_angle(vec)
    assert np.isclose(ang, np.pi / 4.0)


def test_get_rotation_info():
    # check case where all values are provided
    center_row_known = 100
    center_col_known = 200
    vec_known = np.asarray([1, 0])
    (rot_mat_known, ang_known) = com.rot_vec_to_rot_mat_and_angle(vec_known)
    (center_row_found, center_col_found, rot_mat_found, ang_found, vec_found) = com.get_rotation_info(center_row_input=center_row_known, center_col_input=center_col_known, vec_input=vec_known)
    assert np.allclose(rot_mat_known, rot_mat_found)
    assert np.isclose(ang_known, ang_found)
    assert np.isclose(center_row_known, center_row_found)
    assert np.isclose(center_col_known, center_col_found)
    assert np.allclose(vec_known, vec_found)
    # check case where only mask is provided
    file_path = glob_ph1("test_ph1_movie_mini_Anish")[0]
    example_file = io.imread(file_path)
    thresh_img = seg.threshold_array(example_file, 4)
    tissue_mask, wound_mask, _ = seg.isolate_masks(thresh_img, 1)
    mask = seg.make_tissue_mask_robust(tissue_mask, wound_mask)
    center_row_known, center_col_known, vec_known = com.axis_from_mask(mask)
    (rot_mat_known, ang_known) = com.rot_vec_to_rot_mat_and_angle(vec_known)
    (center_row_found, center_col_found, rot_mat_found, ang_found, vec_found) = com.get_rotation_info(mask=mask)
    assert np.allclose(rot_mat_known, rot_mat_found)
    assert np.isclose(ang_known, ang_found)
    assert np.isclose(center_row_known, center_row_found)
    assert np.isclose(center_col_known, center_col_found)
    assert np.allclose(vec_known, vec_found)
    center_row_known = 10
    (center_row_found, center_col_found, rot_mat_found, ang_found, vec_found) = com.get_rotation_info(mask=mask, center_row_input=center_row_known)
    assert np.allclose(rot_mat_known, rot_mat_found)
    assert np.isclose(ang_known, ang_found)
    assert np.isclose(center_row_known, center_row_found)
    assert np.isclose(center_col_known, center_col_found)
    assert np.allclose(vec_known, vec_found)


def test_rot_image():
    mask = np.zeros((100, 100))
    for kk in range(10, 50):
        mask[kk, kk + 20:kk + 30] = 1
    center_row, center_col, vec = com.axis_from_mask(mask)
    (_, ang) = com.rot_vec_to_rot_mat_and_angle(vec)
    new_img = com.rot_image(mask, center_row, center_col, ang)
    new_center_row, new_center_col, new_vec = com.axis_from_mask(new_img)
    assert np.isclose(center_row, new_center_row, atol=2)
    assert np.isclose(center_col, new_center_col, atol=2)
    assert np.allclose(new_vec, np.asarray([0, 1]))


def test_rotate_points():
    row_pts = []
    col_pts = []
    mask = np.zeros((100, 100))
    for kk in range(10, 50):
        mask[kk, kk + 20:kk + 30] = 1
        row_pts.append(kk)
        col_pts.append(kk + 25)
    center_row, center_col, vec = com.axis_from_mask(mask)
    (rot_mat, ang) = com.rot_vec_to_rot_mat_and_angle(vec)
    row_pts = np.asarray(row_pts)
    col_pts = np.asarray(col_pts)
    new_row_pts, new_col_pts = com.rotate_points(row_pts, col_pts, rot_mat, center_row, center_col)
    new_img = com.rot_image(mask, center_row, center_col, ang)
    # plt.figure()
    # plt.imshow(new_img)
    # plt.plot(new_col_pts, new_row_pts, "r.")
    # plt.figure()
    # plt.imshow(mask)
    # plt.plot(col_pts, row_pts, "r.")
    vals = np.nonzero(new_img)
    min_col = np.min(vals[1])
    max_col = np.max(vals[1])
    mean_col = np.mean(vals[1])
    assert np.allclose(new_row_pts, center_row * np.ones(new_row_pts.shape[0]), atol=1)
    assert np.isclose(min_col, np.min(new_col_pts), atol=5)
    assert np.isclose(max_col, np.max(new_col_pts), atol=5)
    assert np.isclose(mean_col, np.mean(new_col_pts), atol=5)


def test_invert_rot_mat():
    mat = np.asarray([[np.sqrt(2)/2, -np.sqrt(2)/2], [np.sqrt(2)/2, np.sqrt(2)/2]])
    mat_inv = com.invert_rot_mat(mat)
    mult = np.dot(mat, mat_inv)
    assert np.allclose(mult, np.eye(2))


def test_get_tissue_width():
    tissue_mask_robust = np.zeros((100, 100))
    tissue_mask_robust[40:61, 30:81] = 1
    tissue_width, pt1_0_orig, pt1_1_orig, pt2_0_orig, pt2_1_orig = com.get_tissue_width(tissue_mask_robust)
    assert np.isclose(tissue_width, 20.0)
    assert np.isclose(pt1_0_orig, 55, 1.0)
    assert np.isclose(pt2_0_orig, 55, 1.0)
    assert np.isclose(pt1_1_orig, 40, 1.0)
    assert np.isclose(pt2_1_orig, 60, 1.0)
    # rotate fake mask run test
    tissue_mask_robust = np.zeros((100, 100))
    tissue_mask_robust[30:81, 40:61] = 1
    tissue_width, pt1_0_orig, pt1_1_orig, pt2_0_orig, pt2_1_orig = com.get_tissue_width(tissue_mask_robust)
    assert np.isclose(tissue_width, 20.0)
    assert np.isclose(pt1_1_orig, 55, 1.0)
    assert np.isclose(pt2_1_orig, 55, 1.0)
    assert np.isclose(pt1_0_orig, 40, 1.0)
    assert np.isclose(pt2_0_orig, 60, 1.0)


def test_get_tissue_width_zoom():
    tissue_mask = np.zeros((500, 500))
    tissue_mask[50:450, 0:500] = 1
    wound_mask = np.zeros((500, 500))
    tissue_width, pt1_0_orig, pt1_1_orig, pt2_0_orig, pt2_1_orig = com.get_tissue_width_zoom(tissue_mask, wound_mask)
    assert np.isclose(tissue_width, 400, 20)
    dist = ((pt1_0_orig - pt2_0_orig) ** 2.0 + (pt1_1_orig - pt2_1_orig) ** 2.0) ** 0.5
    assert np.isclose(tissue_width, dist, 5)
    # example where tissue regions are less than 2
    tissue_mask = np.zeros((500, 500))
    wound_mask = np.zeros((500, 500))
    tissue_width, pt1_0_orig, pt1_1_orig, pt2_0_orig, pt2_1_orig = com.get_tissue_width_zoom(tissue_mask, wound_mask)
    assert np.isclose(tissue_width, 0.0)
    assert np.isclose(pt1_0_orig, 0.0)
    assert np.isclose(pt1_1_orig, 0.0)
    assert np.isclose(pt2_0_orig, 0.0)
    assert np.isclose(pt2_1_orig, 0.0)


def test_compute_dist_line_pt():
    pt0 = 0
    pt1 = 0
    line = np.random.random((100, 2))
    dists = com.compute_dist_line_pt(pt0, pt1, line)
    assert np.allclose(dists, (line[:, 0] ** 2.0 + line[:, 1] ** 2.0) ** 0.5)


def test_tissue_parameters():
    folder_path = example_path("test_ph1_mini_movie")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    img_list = [img_list[0]]
    threshold_function_idx = 3
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, _ = seg.mask_all(thresholded_list, threshold_function_idx)
    width, area, kappa_1, kappa_2, pt1_0, pt1_1, pt2_0, pt2_1, tissue_contour = com.tissue_parameters_zoom(tissue_mask_list[0], wound_mask_list[0])
    tissue_robust = seg.make_tissue_mask_robust(tissue_mask_list[0], wound_mask_list[0])
    regions_all = seg.get_region_props(tissue_robust)
    region = seg.get_largest_regions(regions_all, 1)[0]
    _, axis_major_length, axis_minor_length, centroid_row, centroid_col, _, _, orientation, perimeter = seg.extract_region_props(region)
    # contour_clipped = com.clip_contour(tissue_contour, centroid_row, centroid_col, orientation, axis_major_length, axis_minor_length)
    # import matplotlib.pyplot as plt
    # plt.imshow(img_list[0])
    # plt.plot(tissue_contour[:, 1], tissue_contour[:, 0], 'r-o')
    # plt.plot(contour_clipped[:, 1], contour_clipped[:, 0], 'c-.')
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
    assert tissue_contour.shape[0] > 0
    # import matplotlib.pyplot as plt
    # plt.imshow(img_list[0])
    # plt.plot(tissue_contour[:, 1], tissue_contour[:, 0], 'r-o')
    # plt.plot(pt1_1, pt1_0, 'bo')
    # plt.plot(pt2_1, pt2_0, 'go')
    tissue_mask = np.zeros((100, 200))
    wound_mask = np.zeros((100, 200))
    tissue_width, area, kappa_1, kappa_2, pt1_1_orig, pt1_0_orig, pt2_1_orig, pt2_0_orig, tissue_contour = com.tissue_parameters_zoom(tissue_mask, wound_mask)
    assert np.isclose(tissue_width, 0)
    assert np.isclose(area, 0.0)
    assert np.isclose(kappa_1, 0.0)
    assert np.isclose(kappa_2, 0.0)
    assert np.isclose(pt1_1_orig, 0.0)
    assert np.isclose(pt1_0_orig, 0.0)
    assert np.isclose(pt2_0_orig, 0.0)
    assert np.isclose(pt2_1_orig, 0.0)
    assert tissue_contour is None


def test_contour_all_and_wound_parameters_all_and_tissue_parameters_all():
    folder_path = example_path("test_mini_movie")
    path_dict = ia.input_info_to_input_paths(folder_path)
    folder_path = path_dict["brightfield_images_path"]
    tiff_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = 1
    thresholded_list = seg.threshold_all(tiff_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, wound_region_list = seg.mask_all(thresholded_list, 1)
    contour_list = seg.contour_all(wound_mask_list)
    area_list, axis_major_length_list, axis_minor_length_list, perimeter_list = com.wound_parameters_all(tiff_list[0], contour_list)
    # area_list, axis_major_length_list, axis_minor_length_list = com.wound_parameters_all(wound_region_list)
    zoom_fcn_idx = 1
    tissue_parameter_list = com.tissue_parameters_all(tissue_mask_list, wound_mask_list, zoom_fcn_idx)
    assert len(tissue_mask_list) == 5
    assert len(contour_list) == 5
    assert len(wound_mask_list) == 5
    assert len(area_list) == 5
    assert len(axis_major_length_list) == 5
    assert len(axis_minor_length_list) == 5
    assert len(perimeter_list) == 5
    assert len(tissue_parameter_list) == 5
    assert np.max(area_list) < 512 * 512
    assert np.min(area_list) >= 0
    assert np.max(axis_major_length_list) < 512
    assert np.min(axis_major_length_list) >= 0
    assert np.max(axis_minor_length_list) < 512
    assert np.min(axis_minor_length_list) >= 0
    assert np.max(perimeter_list) < 512*4
    assert np.min(perimeter_list) >= 0
    for kk in range(0, 5):
        assert axis_major_length_list[kk] >= axis_minor_length_list[kk]


def test_get_tissue_width_rotated():
    tissue_mask_robust = np.zeros((100, 100))
    tissue_mask_robust[40:61, 30:81] = 1
    tissue_mask_robust = com.rot_image(tissue_mask_robust, 50, 50, np.pi / 4.0)
    tissue_width, pt1_0, pt1_1, pt2_0, pt2_1 = com.get_tissue_width(tissue_mask_robust)
    assert np.isclose(tissue_width, 20.0, 3.0)
    dist = ((pt1_0 - pt2_0) ** 2.0 + (pt1_1 - pt2_1) ** 2.0) ** 0.5
    assert np.isclose(dist, 20, 3)


def test_get_tissue_width_real():
    file_path = glob_ph1("test_ph1_movie_mini_Anish")[0]
    example_file = io.imread(file_path)
    thresh_img = seg.threshold_array(example_file, 4)
    tissue_mask, wound_mask, _ = seg.isolate_masks(thresh_img, 1)
    tissue_mask_robust = seg.make_tissue_mask_robust(tissue_mask, wound_mask)
    tissue_width, pt1_0_orig, pt1_1_orig, pt2_0_orig, pt2_1_orig = com.get_tissue_width(tissue_mask_robust)
    dist = ((pt1_0_orig - pt2_0_orig) ** 2.0 + (pt1_1_orig - pt2_1_orig) ** 2.0) ** 0.5
    assert np.isclose(dist, tissue_width)


def test_wound_parameters_all_none():
    img = np.zeros((100, 100))
    contour_list = [None, None, None]
    area_list, maj_list, min_list, peri_list = com.wound_parameters_all(img, contour_list)
    for kk in range(0, 3):
        assert area_list[kk] == 0
        maj_list[kk] == 0
        min_list[kk] == 0
        peri_list[kk] == 0


def test_tissue_parameters_anish():
    folder_path = example_path("test_ph1_movie_mini_Anish")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    img_list = [img_list[0]]
    threshold_function_idx = 4
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, _ = seg.mask_all(thresholded_list, threshold_function_idx)
    width, area, kappa_1, kappa_2, pt1_0, pt1_1, pt2_0, pt2_1, tissue_contour = com.tissue_parameters(tissue_mask_list[0], wound_mask_list[0])
    tissue_robust = seg.make_tissue_mask_robust(tissue_mask_list[0], wound_mask_list[0])
    regions_all = seg.get_region_props(tissue_robust)
    region = seg.get_largest_regions(regions_all, 1)[0]
    _, axis_major_length, axis_minor_length, centroid_row, centroid_col, _, _, orientation,perimeter = seg.extract_region_props(region)
    # contour_clipped = ia.clip_contour(tissue_contour, centroid_row, centroid_col, orientation, axis_major_length, axis_minor_length)
    # contour_clipped_2 = ia.clip_contour(contour_clipped, centroid_row, centroid_col, orientation - np.pi / 2.0, axis_major_length, axis_minor_length)
    import matplotlib.pyplot as plt
    plt.imshow(img_list[0])
    plt.plot(tissue_contour[:, 1], tissue_contour[:, 0], 'r-o')
    # plt.plot(contour_clipped_2[:, 1], contour_clipped_2[:, 0], 'c-.')
    plt.plot(pt1_0, pt1_1, 'bo')
    plt.plot(pt2_0, pt2_1, 'go')
    assert width > 0
    assert area > 0
    assert kappa_1 < 1
    assert kappa_2 < 1
    assert pt1_0 > 0 and pt1_0 < img_list[0].shape[1]
    assert pt2_0 > 0 and pt2_0 < img_list[0].shape[1]
    assert pt1_1 > 0 and pt1_1 < img_list[0].shape[0]
    assert pt2_1 > 0 and pt2_1 < img_list[0].shape[0]
    assert tissue_contour.shape[0] > 100


def test_check_broken_tissue():
    tissue_mask_orig = np.zeros((100, 100))
    tissue_mask_orig[20:75, 20:75] = 1
    tissue_mask = np.zeros((100, 100))
    tissue_mask[50:55, 0:55] = 1
    is_broken = com.check_broken_tissue(tissue_mask, tissue_mask_orig)
    assert is_broken is True


def test_split_into_four_corners_with_pillars():
    tissue_mask = np.zeros((8,8))
    tissue_mask[2:6,2:6] = 1
    p0_mask = np.ones((8,8),dtype=bool)
    p0_mask[1,1] = 0
    p1_mask = np.ones((8,8),dtype=bool)
    p1_mask[-2,-2] = 0
    p2_mask = np.ones((8,8),dtype=bool)
    p2_mask[1,-2] = 0
    p3_mask = np.ones((8,8),dtype=bool)
    p3_mask[-2,1] = 0
    pillar_mask_list = [p0_mask,p1_mask,p2_mask,p3_mask]
    for pil_ind,pillar_mask in enumerate(pillar_mask_list):
        pillar_mask_list[pil_ind] = seg.invert_mask(pillar_mask).astype(np.uint8)
    tissue_quarter_masks = com.split_into_four_corners_with_pillars(
        tissue_mask,pillar_mask_list,0
    )
    assert len(tissue_quarter_masks) == 4
    assert isinstance(tissue_quarter_masks[0],np.ndarray)


def test_obtain_tissue_quarters_area():
    tissue_mask = np.zeros((8,8))
    tissue_mask[2:6,2:6] = 1
    p0_mask = np.ones((8,8),dtype=bool)
    p0_mask[1,1] = 0
    p1_mask = np.ones((8,8),dtype=bool)
    p1_mask[-2,-2] = 0
    p2_mask = np.ones((8,8),dtype=bool)
    p2_mask[1,-2] = 0
    p3_mask = np.ones((8,8),dtype=bool)
    p3_mask[-2,1] = 0
    pillar_mask_list = [p0_mask,p1_mask,p2_mask,p3_mask]
    for pil_ind,pillar_mask in enumerate(pillar_mask_list):
        pillar_mask_list[pil_ind] = seg.invert_mask(pillar_mask).astype(np.uint8)
    Qlist,tissue_quarter_masks=com.obtain_tissue_quarters_area(
        tissue_mask,pillar_mask_list,0
    )
    assert isinstance(Qlist,list)
    assert isinstance(tissue_quarter_masks[0],np.ndarray)


def test_binary_mask_IOU():
    mask1 = np.zeros((100, 100))
    mask2 = np.ones((100, 100))
    iou = com.binary_mask_IOU(mask1, mask2)
    assert iou == 0
    mask1 = np.ones((100, 100))
    mask2 = np.ones((100, 100))
    iou = com.binary_mask_IOU(mask1, mask2)
    assert iou == 1
    mask1 = np.ones((100, 100))
    mask2 = np.ones((100, 100))
    mask2[0:50, :] = 0
    iou = com.binary_mask_IOU(mask1, mask2)
    assert iou > 0 and iou < 1


def test_check_broken_tissue_zoom():
    is_broken_list = []
    folder_path = example_path("test_zoom_is_broken")
    input_dict, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = seg.select_threshold_function(input_dict, False, False, True, False)
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, _ = seg.mask_all(thresholded_list, threshold_function_idx)
    for kk in range(0, int(len(img_list) / 2)):
        ix_orig = kk * 2
        ix_consider = kk * 2 + 1
        tissue_mask_orig = tissue_mask_list[ix_orig]
        wound_mask_orig = wound_mask_list[ix_orig]
        is_broken = com.check_broken_tissue_zoom(tissue_mask_list[ix_consider], wound_mask_list[ix_consider], tissue_mask_orig, wound_mask_orig)
        is_broken_list.append(is_broken)
    ground_truth = [True, False, True, False]
    for kk in range(0, len(ground_truth)):
        assert ground_truth[kk] is is_broken_list[kk]


def test_check_broken_tissue_all():
    tissue_mask_list = []
    wound_mask_list = []
    for kk in range(0, 3):
        tissue_mask = np.zeros((10, 10))
        tissue_mask[3:7, 3:7] = 1
        tissue_mask_list.append(tissue_mask)
        wound_mask_list.append(np.zeros((10, 10)))
    is_broken_list = com.check_broken_tissue_all(tissue_mask_list, wound_mask_list)
    print(is_broken_list)
    for bb in is_broken_list:
        assert bb is False
    is_broken_list = com.check_broken_tissue_all(tissue_mask_list, wound_mask_list, False, 2)
    print(is_broken_list)
    for bb in is_broken_list:
        assert bb is False


def test_check_broken_tissue_with_pillars_no_tissue():
    tm = np.zeros((20,20))
    pillar_mask_list = [np.zeros((20,20))]
    is_broken = com.check_broken_tissue_with_pillars(tm,pillar_mask_list)
    assert is_broken is True


def test_check_broken_tissue_with_pillars_tiny_tissue():
    tm = np.zeros((100,100))
    tm[50,50] = 1
    pillar_mask_list = [np.zeros((100,100))]
    is_broken = com.check_broken_tissue_with_pillars(tm,pillar_mask_list)
    assert is_broken is True


def test_is_broken_example():
    folder_path = example_path("test_ph1_mini_movie_broken")
    input_dict, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = seg.select_threshold_function(input_dict, False, False, True, False)
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, _, _ = seg.mask_all(thresholded_list, threshold_function_idx)
    is_broken_list = com.check_broken_tissue_all(tissue_mask_list, [], True, zoom_type=2, pillar_mask_list=[])
    assert is_broken_list[0] is False
    assert is_broken_list[1] is True
    assert is_broken_list[2] is True
    assert is_broken_list[3] is True


# version that is more for debugging method than for testing 
# def test_check_broken_tissue_zoom():
#     import matplotlib.pyplot as plt
#     is_broken_list = []
#     iou_list = []
#     folder_path = example_path("test_zoom_orig_compare")
#     input_dict, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
#     folder_path = input_path_dict["ph1_images_path"]
#     img_list = ia.read_all_tiff(folder_path)
#     path_list = ia.image_folder_to_path_list(folder_path)
#     threshold_function_idx = seg.select_threshold_function(input_dict, False, False, True)
#     thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
#     tissue_mask_list, wound_mask_list, _ = seg.mask_all(thresholded_list, threshold_function_idx)
#     for kk in range(0, len(img_list)):
#         ix_orig = kk - kk % 3
#         tissue_mask_orig = tissue_mask_list[ix_orig]
#         wound_mask_orig = wound_mask_list[ix_orig]
#         ti = str(path_list[kk]).split("/")[-1]
#         is_broken, iou_masks = com.check_broken_tissue_zoom(tissue_mask_list[kk], wound_mask_list[kk], tissue_mask_orig, wound_mask_orig)
#         is_broken_list.append(is_broken)
#         print(ti, is_broken)
#         plt.figure()
#         plt.imshow(tissue_mask_list[kk])
#         plt.title(ti + "broken:" + str(is_broken) + " iou: %0.2f" % (iou_masks))
#         iou_list.append(iou_masks)
#     aa = 44


# def test_check_broken_tissue_zoom():
#     import matplotlib.pyplot as plt
#     folder_path = example_path("test_zoom")
#     input_dict, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
#     folder_path = input_path_dict["ph1_images_path"]
#     img_list = ia.read_all_tiff(folder_path)
#     path_list = ia.image_folder_to_path_list(folder_path)
#     threshold_function_idx = seg.select_threshold_function(input_dict, False, False, True)
#     thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
#     tissue_mask_list, _, _ = seg.mask_all(thresholded_list, threshold_function_idx)
#     # check individual tissues
#     is_broken_list = []
#     for kk in range(0, len(img_list)):
#         ti = str(path_list[kk]).split("/")[-1]
#         is_broken = com.check_broken_tissue_zoom(tissue_mask_list[kk])
#         is_broken_list.append(is_broken)
#         print(ti, is_broken)
#         plt.figure()
#         plt.imshow(tissue_mask_list[kk])
#         plt.title(ti + "broken:" + str(is_broken))
#     aa = 44


def test_is_broken_anish():
    folder_path = example_path("test_alt_broken_Anish")
    input_dict, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = seg.select_threshold_function(input_dict, False, False, True, False)
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, _, _ = seg.mask_all(thresholded_list, threshold_function_idx)
    pillar_mask_list,_ = seg.get_pillar_mask_list(img_list[0],4)
    is_broken_list = com.check_broken_tissue_all(tissue_mask_list,zoom_type=2,pillar_mask_list=pillar_mask_list)
    for kk in range(0, len(is_broken_list)):
        assert is_broken_list[kk] is True


def test_is_broken_alt():
    folder_path = example_path("test_alt_broken")
    input_dict, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["brightfield_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = seg.select_threshold_function(input_dict, True, False, False, False)
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, _, _ = seg.mask_all(thresholded_list, threshold_function_idx)
    tissue_mask_list = [tissue_mask_list[3]]
    is_broken_list = com.check_broken_tissue_all(tissue_mask_list)
    for kk in range(0, len(is_broken_list)):
        assert is_broken_list[kk] is True


def test_is_broken_unbroken_example():
    folder_path = example_path("test_ph1_mini_movie")
    input_dict, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = seg.select_threshold_function(input_dict, False, False, True, False)
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, _, _ = seg.mask_all(thresholded_list, threshold_function_idx)
    is_broken_list = com.check_broken_tissue_all(tissue_mask_list)
    for bb in is_broken_list:
        assert bb is False


def test_shrink_bounding_box():
    (min_row, min_col, max_row, max_col) = (100, 50, 140, 130)
    shrink_factor = 0.5
    (min_row_new, min_col_new, max_row_new, max_col_new) = com.shrink_bounding_box(min_row, min_col, max_row, max_col, shrink_factor)
    assert min_row_new == 100 + 10
    assert min_col_new == 50 + 20
    assert max_row_new == 140 - 10
    assert max_col_new == 130 - 20


def test_check_inside_box():
    array = np.zeros((20, 20))
    array[8:12, 8:12] = 1
    region = seg.get_region_props(array)[0]
    bbox = (5, 5, 15, 15)
    assert com.check_inside_box(region, bbox, bbox) is True
    bbox = (5, 10, 15, 11)
    assert com.check_inside_box(region, bbox, bbox) is False


def test_check_wound_closed_is_open():
    folder_path = example_path("test_ph1_mini_movie")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = 3
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, _, wound_region_list = seg.mask_all(thresholded_list, threshold_function_idx)
    check_closed = com.check_wound_closed_zoom(tissue_mask_list[0], wound_region_list[0])
    assert check_closed is False


def test_check_wound_closed_is_closed():
    # check a closed example
    folder_path = example_path("test_mini_movie_closing")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["brightfield_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = 1
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, _, wound_region_list = seg.mask_all(thresholded_list, threshold_function_idx)
    check_closed = com.check_wound_closed_zoom(tissue_mask_list[5], wound_region_list[5])
    assert check_closed is True


def test_check_wound_closed_zoom_none():
    tissue_mask = np.zeros((100, 100))
    wound_region = None
    is_closed = com.check_wound_closed_zoom(tissue_mask, wound_region)
    assert is_closed


def test_check_wound_closed_all():
    folder_path = example_path("test_mini_movie_closing")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["brightfield_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = 1
    zoom_fcn_idx = 1
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, _, wound_region_list = seg.mask_all(thresholded_list, threshold_function_idx)
    check_closed_list = com.check_wound_closed_all(tissue_mask_list, wound_region_list, zoom_fcn_idx)
    for kk in range(0, 3):
        assert check_closed_list[kk] is False
    for kk in range(3, 7):
        assert check_closed_list[kk] is True


def test_check_wound_closed_all_Anish():
    folder_path = example_path("test_ph1_mini_movie_closing_Ansih")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    threshold_function_idx = 4
    zoom_fcn_idx = 2
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, _, wound_region_list = seg.mask_all(thresholded_list, threshold_function_idx)
    check_closed_list = com.check_wound_closed_all(tissue_mask_list, wound_region_list, zoom_fcn_idx)
    for kk in range(0, 3):
        assert check_closed_list[kk] is False
    for kk in range(3, 6):
        assert check_closed_list[kk] is True


def test_mask_to_area():
    rad_1 = 5
    disk_1 = morphology.disk(rad_1, dtype=bool)
    region_props = seg.get_region_props(disk_1)
    region = region_props[0]
    coords = [seg.extract_region_props(region)[5]]
    mask = seg.coords_to_mask(coords, disk_1)
    pix_to_microns = 1
    area = com.mask_to_area(mask, pix_to_microns)
    assert area == np.sum(disk_1)


# def test_tissue_parameters_anish_challenge():
#     folder_path = example_path("test_ph1_movie_mini_Anish")
#     _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
#     folder_path = input_path_dict["ph1_images_path"]
#     img_list_all = ia.read_all_tiff(folder_path)
#     for kk in range(0, len(img_list_all)):
#         img_list = [img_list_all[kk]]
#         threshold_function_idx = 4
#         thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
#         tissue_mask_list, wound_mask_list, _ = seg.mask_all(thresholded_list, threshold_function_idx)
#         width, area, kappa_1, kappa_2, pt1_0, pt1_1, pt2_0, pt2_1, tissue_contour = com.tissue_parameters(tissue_mask_list[0], wound_mask_list[0])
#         tissue_robust = seg.make_tissue_mask_robust(tissue_mask_list[0], wound_mask_list[0])
#         regions_all = seg.get_region_props(tissue_robust)
#         region = seg.get_largest_regions(regions_all, 1)[0]
#         _, axis_major_length, axis_minor_length, centroid_row, centroid_col, _, _, orientation = seg.extract_region_props(region)
#         contour_clipped = ia.clip_contour(tissue_contour, centroid_row, centroid_col, orientation, axis_major_length, axis_minor_length)
#         contour_clipped_2 = ia.clip_contour(contour_clipped, centroid_row, centroid_col, orientation - np.pi / 2.0, axis_major_length, axis_minor_length)
#         import matplotlib.pyplot as plt
#         plt.figure()
#         plt.imshow(img_list_all[kk])
#         plt.plot(tissue_contour[:, 1], tissue_contour[:, 0],'r-o')
#         plt.plot(contour_clipped_2[:, 1], contour_clipped_2[:, 0],'c-.')
#         plt.plot(pt1_0, pt1_1, 'bo')
#         plt.plot(pt2_0, pt2_1, 'go')


def test_get_contour_distance_across():
    rad = 50
    disk = morphology.disk(rad, bool)
    array = np.zeros((rad * 5, rad * 5))
    array[rad:rad + disk.shape[0], rad:rad + disk.shape[1]] = disk
    contour = seg.mask_to_contour(array)
    num_pts_contour = contour.shape[0]
    tolerence_check = 0.1
    c_idx = 0
    include_idx = list(range(0, contour.shape[0]))
    distance, ix_opposite = com.get_contour_distance_across(c_idx, contour, num_pts_contour, include_idx, tolerence_check)
    assert distance < rad * 2.0
    assert distance > rad * 2.0 * 0.9
    assert ix_opposite > num_pts_contour * 0.25
    assert ix_opposite < num_pts_contour * 0.75


def test_get_contour_distance_across_all():
    val1 = 30
    val2 = 35
    array = np.zeros((val2 * 5, val2 * 5))
    array[val2:val2 * 2, val1:val1 * 2] = 1
    contour = seg.mask_to_contour(array)
    regions_all = seg.get_region_props(array)
    region = seg.get_largest_regions(regions_all, 1)[0]
    _, tissue_axis_major_length, tissue_axis_minor_length, centroid_row, centroid_col, _, _, _, _ = seg.extract_region_props(region)
    include_idx = com.include_points_contour(contour, centroid_row, centroid_col, tissue_axis_major_length, tissue_axis_minor_length)
    distance_all, _ = com.get_contour_distance_across_all(contour, include_idx)
    assert np.min(distance_all) < val1 * 1.05
    assert np.max(distance_all[distance_all < math.inf]) < val2 * 1.05
    # sum_all = []
    # for kk in range(0, len(ix_all)):
    #     sum_all.append(kk + ix_all[kk])
    # assert np.min(sum_all) > contour.shape[0] * 0.1


def test_include_points_contour():
    val1 = 100
    val2 = 200
    array = np.zeros((val2 * 10, val2 * 10))
    array[val2:val2 * 2, val1:val1 * 2] = 1
    contour = seg.mask_to_contour(array)
    regions_all = seg.get_region_props(array)
    region = seg.get_largest_regions(regions_all, 1)[0]
    _, tissue_axis_major_length, tissue_axis_minor_length, centroid_row, centroid_col, _, _, _, _ = seg.extract_region_props(region)
    include_idx = com.include_points_contour(contour, centroid_row, centroid_col, tissue_axis_major_length, tissue_axis_minor_length)
    include_idx = np.asarray(include_idx)
    # import matplotlib.pyplot as plt
    # plt.plot(contour[:, 0], contour[:, 1], 'r-')
    # plt.plot(contour[include_idx, 0], contour[include_idx, 1], 'co')
    for kk in range(0, include_idx.shape[0]):
        di = seg.compute_distance(contour[include_idx[kk], 0], contour[include_idx[kk], 1], centroid_row, centroid_col)
        assert di < 0.25 * (tissue_axis_major_length + tissue_axis_minor_length)


def test_get_contour_width():
    val1 = 100
    val2 = 200
    array = np.zeros((val2 * 5, val2 * 5))
    array[val2:val2 * 2, val1:val1 * 2] = 1
    contour = seg.mask_to_contour(array)
    regions_all = seg.get_region_props(array)
    region = seg.get_largest_regions(regions_all, 1)[0]
    _, tissue_axis_major_length, tissue_axis_minor_length, centroid_row, centroid_col, _, _, orientation, perimeter = seg.extract_region_props(region)
    width, idx_a, idx_b = com.get_contour_width(contour, centroid_row, centroid_col, tissue_axis_major_length, tissue_axis_minor_length, orientation)
    assert width < val1
    assert width > val1 * 0.9
    p0a = contour[idx_a, 0]
    p1a = contour[idx_a, 1]
    p0b = contour[idx_b, 0]
    p1b = contour[idx_b, 1]
    dist = ((p0a - p0b)**2.0 + (p1a - p1b)**2.0) ** 0.5
    assert pytest.approx(dist, .1) == width


def test_select_zoom_function():
    dict = {"zoom_type": 1}
    val = com.select_zoom_function(dict)
    assert val == 1
    dict = {"zoom_type": 2}
    val = com.select_zoom_function(dict)
    assert val == 2
    dict = {"zoom_type": 3}
    val = com.select_zoom_function(dict)
    assert val == 3


def test_compute_linear_healing_rate():
    area_list = [100, 400, 100, 25]
    perimeter_list = [40, 80, 40, 20]
    known = [0, -300/80, 300/40, 75/20]
    found = com.compute_linear_healing_rate(area_list, perimeter_list, 1.0)
    assert np.allclose(known,found)


def test_compute_dark_pixels_ratio_at_mask_edge():
    dim = 100
    img_arr = np.ones((dim,dim))
    img_arr[30:50,30:50] = 0
    binary_mask = np.zeros((dim,dim))
    binary_mask[20:60,20:60] = 1
    found_dark_pixel_ratio1 = com.compute_dark_pixels_ratio_at_mask_edge(binary_mask,img_arr)
    assert found_dark_pixel_ratio1 < 0.05
    binary_mask2 = np.zeros((dim,dim))
    binary_mask2[30:50,30:50] = 1
    found_dark_pixel_ratio2 = com.compute_dark_pixels_ratio_at_mask_edge(binary_mask2,img_arr)
    assert found_dark_pixel_ratio2 > 0.50
