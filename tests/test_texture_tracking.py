import numpy as np
from pathlib import Path
from woundcompute import image_analysis as ia
from woundcompute import segmentation as seg
from woundcompute import texture_tracking as tt


def files_path():
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    return data_path


def example_path(example_name):
    data_path = files_path()
    example_path = data_path.joinpath(example_name).resolve()
    return example_path


def test_get_tracking_param_dicts():
    feature_params, lk_params = tt.get_tracking_param_dicts()
    assert feature_params["maxCorners"] == 10000
    assert feature_params["qualityLevel"] == 0.01
    assert feature_params["minDistance"] == 10
    assert feature_params["blockSize"] == 10
    assert lk_params["winSize"][0] == 100
    assert lk_params["winSize"][1] == 100
    assert lk_params["maxLevel"] == 10
    assert lk_params["criteria"][1] == 10
    assert lk_params["criteria"][2] == 0.03


def test_get_tracking_param_dicts_pillar():
    feature_params, lk_params = tt.get_tracking_param_dicts_pillar()
    assert feature_params["maxCorners"] == 100
    assert feature_params["qualityLevel"] == 0.01
    assert feature_params["minDistance"] == 3
    assert feature_params["blockSize"] == 3
    assert lk_params["winSize"][0] == 100
    assert lk_params["winSize"][1] == 100
    assert lk_params["maxLevel"] == 10
    assert lk_params["criteria"][1] == 10
    assert lk_params["criteria"][2] == 0.03


def test_uint16_to_uint8():
    array_8 = np.random.randint(0, 255, (5, 5)).astype(np.uint8)
    array_8[0, 0] = 0
    array_8[1, 0] = 255
    array_16 = array_8.astype(np.uint16) * 100
    found = tt.uint16_to_uint8(array_16)
    assert np.allclose(array_8, found)


def test_uint16_to_uint8_all():
    folder_path = example_path("test_mini_movie")
    path_dict = ia.input_info_to_input_paths(folder_path)
    folder_path = path_dict["brightfield_images_path"]
    tiff_list = ia.read_all_tiff(folder_path)
    uint8_list = tt.uint16_to_uint8_all(tiff_list)
    for img in uint8_list:
        assert img.dtype is np.dtype('uint8')


def test_get_unique():
    numbers = [1, 1, 2, 3, 3, 3, 3, 4, 5]
    list_unique = tt.get_unique(numbers)
    assert len(list_unique) == 5
    assert 1 in list_unique
    assert 2 in list_unique
    assert 3 in list_unique
    assert 4 in list_unique
    assert 5 in list_unique
    numbers = [4, 4, 4, 3, 2, 2, 5, 1, 1, 1, 1]
    list_unique = tt.get_unique(numbers)
    assert len(list_unique) == 5
    assert 1 in list_unique
    assert 2 in list_unique
    assert 3 in list_unique
    assert 4 in list_unique
    assert 5 in list_unique


def test_get_order_track():
    len_img_list = 3
    is_forward = True
    order_list = tt.get_order_track(len_img_list, is_forward)
    for kk in range(0, len_img_list):
        assert order_list[kk] is kk
    is_forward = False
    order_list = tt.get_order_track(len_img_list, is_forward)
    for kk in range(0, len_img_list):
        assert order_list[kk] is len_img_list - kk - 1


def test_bool_to_uint8():
    arr_bool = np.random.random((10, 10)) > 0.5
    arr_uint8 = tt.bool_to_uint8(arr_bool)
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
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, _, _ = seg.mask_all(thresholded_list, threshold_function_idx)
    tissue_mask = tissue_mask_list[0]
    # wound_mask = wound_mask_list[0]
    # wound_contour = ia.mask_to_contour(wound_mask)
    img_uint8 = tt.uint16_to_uint8(img_list[0])
    feature_params, lk_params = tt.get_tracking_param_dicts()
    track_points_0 = tt.mask_to_track_points(img_uint8, tissue_mask, feature_params)
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
    img_uint8_0 = tt.uint16_to_uint8(img_list_orig[0])
    img_uint8_1 = tt.uint16_to_uint8(img_list_orig[1])
    track_points_1 = tt.track_one_step(img_uint8_0, img_uint8_1, track_points_0, lk_params)
    assert track_points_0.shape == track_points_1.shape
    diff_0 = np.abs(track_points_0[:, 0, 0] - track_points_1[:, 0, 0])
    diff_1 = np.abs(track_points_0[:, 0, 1] - track_points_1[:, 0, 1])
    window_0 = lk_params["winSize"][0]
    window_1 = lk_params["winSize"][1]
    assert np.max(diff_0) < window_0
    assert np.max(diff_1) < window_1
    img_list_uint8 = tt.uint16_to_uint8_all(img_list_orig)
    order_list = tt.get_order_track(len(img_list_uint8), True)
    tracker_x, tracker_y = tt.track_all_steps(img_list_uint8, tissue_mask, order_list)
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
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, _ = seg.mask_all(thresholded_list, threshold_function_idx)
    tissue_mask = tissue_mask_list[0]
    wound_mask = wound_mask_list[0]
    wound_contour = seg.mask_to_contour(wound_mask)
    frame_0_mask = img_list[0]
    img_list_uint8 = tt.uint16_to_uint8_all(img_list)
    order_list = tt.get_order_track(len(img_list_uint8), True)
    tracker_x, tracker_y = tt.track_all_steps(img_list_uint8, tissue_mask, order_list)
    alpha_assigned = False
    mask_wound_initial, mask_wound_final = tt.wound_mask_from_points(frame_0_mask, tracker_x, tracker_y, wound_contour, alpha_assigned)
    assert mask_wound_initial.shape == mask_wound_final.shape
    assert mask_wound_initial.shape == frame_0_mask.shape
    assert np.max(mask_wound_initial) == 1
    assert np.min(mask_wound_initial) == 0
    assert np.max(mask_wound_final) == 1
    assert np.min(mask_wound_final) == 0
    alpha_assigned = True
    assigned_alpha = 0.015
    mask_wound_initial, mask_wound_final = tt.wound_mask_from_points(frame_0_mask, tracker_x, tracker_y, wound_contour, alpha_assigned, assigned_alpha)
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
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, _ = seg.mask_all(thresholded_list, threshold_function_idx)
    frame_0_mask = tissue_mask_list[0]
    include_reverse = True
    wound_mask = wound_mask_list[0]
    wound_contour = seg.mask_to_contour(wound_mask)
    frame_final_mask, tracker_x, tracker_y, tracker_x_reverse, tracker_y_reverse, wound_area_list, wound_masks_all = tt.perform_tracking(frame_0_mask, img_list, include_reverse, wound_contour)
    include_reverse = False
    frame_final_mask, tracker_x_forward, tracker_y_forward, tracker_x_reverse_forward, tracker_y_reverse_forward, _, _ = tt.perform_tracking(frame_0_mask, img_list, include_reverse, wound_contour)
    assert tracker_x.shape[1] == len(img_list)
    assert tracker_y.shape[1] == len(img_list)
    assert tracker_x_reverse.shape[1] == len(img_list)
    assert tracker_y_reverse.shape[1] == len(img_list)
    assert tracker_x_forward.shape[1] == len(img_list)
    assert tracker_y_forward.shape[1] == len(img_list)
    assert tracker_x_reverse_forward is None
    assert tracker_y_reverse_forward is None
    assert frame_final_mask.shape[0] == frame_0_mask.shape[0]
    assert frame_final_mask.shape[1] == frame_0_mask.shape[1]
    assert len(wound_area_list) == len(img_list)
    assert len(wound_masks_all) == len(img_list)
    include_reverse = False
    _, _, _, tracker_x_reverse, tracker_y_reverse, _, _ = tt.perform_tracking(frame_0_mask, img_list, include_reverse, wound_contour)
    assert tracker_x_reverse is None
    assert tracker_y_reverse is None


def test_wound_areas_from_points():
    folder_path = example_path("test_phi_movie_mini_Anish_tracking")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list_orig = ia.read_all_tiff(folder_path)
    img_list = [img_list_orig[0], img_list_orig[1], img_list_orig[2]]
    threshold_function_idx = 4
    thresholded_list = seg.threshold_all(img_list, threshold_function_idx)
    tissue_mask_list, wound_mask_list, _ = seg.mask_all(thresholded_list, threshold_function_idx)
    tissue_mask = tissue_mask_list[0]
    wound_mask = wound_mask_list[0]
    wound_contour = seg.mask_to_contour(wound_mask)
    frame_0_mask = img_list[0]
    img_list_uint8 = tt.uint16_to_uint8_all(img_list)
    order_list = tt.get_order_track(len(img_list_uint8), True)
    tracker_x, tracker_y = tt.track_all_steps(img_list_uint8, tissue_mask, order_list)
    wound_area_list, wound_masks_all = tt.wound_areas_from_points(frame_0_mask, tracker_x, tracker_y, wound_contour)
    assert len(wound_area_list) == tracker_x.shape[1]
    assert len(wound_masks_all) == tracker_x.shape[1]
    wound_area_list, wound_masks_all = tt.wound_areas_from_points(frame_0_mask, tracker_x, tracker_y, wound_contour, False)
    assert len(wound_area_list) == tracker_x.shape[1]
    assert len(wound_masks_all) == tracker_x.shape[1]


def test_perform_pillar_tracking():
    folder_path = example_path("test_pillar_tracking")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list = ia.read_all_tiff(folder_path)
    # img_list = img_list[0:5]
    threshold_function_idx = 4
    pillar_mask_list = seg.get_pillar_mask_list(img_list[0], threshold_function_idx)
    avg_disp_all_x, avg_disp_all_y = tt.perform_pillar_tracking(pillar_mask_list, img_list)
    assert avg_disp_all_x.shape[0] == len(img_list)
    assert avg_disp_all_y.shape[0] == len(img_list)
    assert avg_disp_all_x.shape[1] == 4
    assert avg_disp_all_y.shape[1] == 4


def test_track_all_steps_pillar():
    folder_path = example_path("test_pillar_tracking")
    _, input_path_dict, _ = ia.input_info_to_dicts(folder_path)
    folder_path = input_path_dict["ph1_images_path"]
    img_list_orig = ia.read_all_tiff(folder_path)
    img_list_uint8 = tt.uint16_to_uint8_all(img_list_orig)
    order_list = tt.get_order_track(len(img_list_uint8), True)
    threshold_function_idx = 4
    pillar_mask_list = seg.get_pillar_mask_list(img_list_orig[0], threshold_function_idx)
    tracker_x, tracker_y = tt.track_all_steps(img_list_uint8, pillar_mask_list[0], order_list, True)
    assert tracker_x.shape[1] == len(img_list_orig)
    assert tracker_y.shape[1] == len(img_list_orig)