import numpy as np
from pathlib import Path
import pytest
from woundcompute import image_analysis as ia
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


def test_get_wound_area_is_broken_is_closed():
    folder_path = example_path("test_ph1_movie_mini_Anish")
    yaml_path = folder_path.joinpath("test_ph1_mini_movie.yaml").resolve()
    input_dict = ia._yml_to_dict(yml_path_file=yaml_path)
    path_dict = ia.input_info_to_output_paths(folder_path, input_dict)
    _, _ = ia.run_all(folder_path)
    output_path = path_dict["segment_ph1_path"]
    wound_area = pp.get_wound_area(output_path)
    is_broken = pp.get_is_broken(output_path)
    is_closed = pp.get_is_closed(output_path)
    assert wound_area.shape[0] == 5
    assert is_broken.shape[0] == 5
    assert is_closed.shape[0] == 5


def test_smooth_with_GPR():
    s = np.ones((10))
    yhat = pp.smooth_with_GPR(s)
    assert np.all(np.isclose(s, yhat))
    s[5] = 10
    yhat = pp.smooth_with_GPR(s)
    assert yhat[5] > 1 and yhat[5] < 10
    s = np.linspace(1, 10, 100)
    s = np.sin(s)
    yhat = pp.smooth_with_GPR(s)
    assert np.all(np.isclose(s, yhat, atol=.0001))


def test_is_broken_smooth():
    is_broken = np.asarray([0, 0, 0, 0, 0])
    is_broken_smooth = pp.is_broken_smooth(is_broken)
    assert np.all(np.isclose(is_broken, is_broken_smooth))
    is_broken = np.asarray([0, 0, 0, 0, 1])
    is_broken_smooth = pp.is_broken_smooth(is_broken)
    assert np.all(np.isclose(is_broken, is_broken_smooth))
    is_broken = np.asarray([0, 0, 1, 0, 0])
    is_broken_smooth = pp.is_broken_smooth(is_broken)
    assert np.all(np.isclose(is_broken_smooth, np.zeros(5)))
    is_broken = np.asarray([0, 0, 1, 1, 1])
    is_broken_smooth = pp.is_broken_smooth(is_broken)
    assert np.all(np.isclose(is_broken, is_broken_smooth))


def test_is_closed_smooth():
    is_closed = np.asarray([0, 0, 0, 0, 0])
    is_closed_smooth = pp.is_closed_smooth(is_closed)
    assert np.all(np.isclose(is_closed, is_closed_smooth))
    is_closed = np.asarray([0, 0, 0, 0, 1])
    is_closed_smooth = pp.is_closed_smooth(is_closed)
    assert np.all(np.isclose(is_closed, is_closed_smooth))
    is_closed = np.asarray([0, 0, 1, 0, 0])
    is_closed_smooth = pp.is_closed_smooth(is_closed)
    assert np.all(np.isclose(is_closed_smooth, np.zeros(5)))
    is_closed = np.asarray([0, 0, 0, 1, 1, 0, 0, 0])
    is_closed_smooth = pp.is_closed_smooth(is_closed)
    assert np.all(np.isclose(is_closed_smooth, np.zeros(8)))
    is_closed = np.asarray([0, 0, 1, 1, 1, 0, 0, 0])
    is_closed_smooth = pp.is_closed_smooth(is_closed)
    assert np.all(np.isclose(is_closed_smooth, is_closed))
    is_closed = np.asarray([0, 0, 0, 1, 1, 1, 1, 1])
    is_closed_smooth = pp.is_closed_smooth(is_closed)
    assert np.all(np.isclose(is_closed_smooth, is_closed))


def test_wound_area_with_is_closed():
    is_closed = np.asarray([0, 0, 0, 0, 0])
    wound_area = np.asarray([100, 100, 100, 100, 100])
    wound_area_new = pp.wound_area_with_is_closed(wound_area, is_closed)
    assert np.all(np.isclose(wound_area, wound_area_new))
    is_closed = np.asarray([0, 0, 0, 0, 1])
    wound_area = np.asarray([100, 100, 100, 100, 100])
    wound_area_gt = np.asarray([100, 100, 100, 100, 0])
    wound_area_new = pp.wound_area_with_is_closed(wound_area, is_closed)
    assert np.all(np.isclose(wound_area_gt, wound_area_new))
    is_closed = np.asarray([1, 1, 1, 1, 1])
    wound_area = np.asarray([100, 100, 100, 100, 100])
    wound_area_gt = np.asarray([0, 0, 0, 0, 0])
    wound_area_new = pp.wound_area_with_is_closed(wound_area, is_closed)
    assert np.all(np.isclose(wound_area_gt, wound_area_new))


def test_run_full_postproc_sequence():
    wound_area = np.asarray([100, 150, 125, 100, 100])
    is_broken = np.asarray([0, 0, 0, 0, 0])
    is_closed = np.asarray([0, 0, 0, 0, 0])
    wound_area_final, is_broken_sm, is_closed_sm, frame_broken, frame_closed = pp.run_full_postproc_sequence(wound_area, is_broken, is_closed)
    assert frame_broken is None
    assert frame_closed is None
    assert np.all(np.isclose(is_broken_sm, is_broken))
    assert np.all(np.isclose(is_closed_sm, is_closed))
    assert wound_area_final.shape[0] == wound_area.shape[0]
    wound_area = np.asarray([100, 150, 125, 100, 100, 100, 100, 100])
    is_broken = np.asarray([0, 0, 0, 0, 0, 1, 1, 1])
    is_closed = np.asarray([0, 0, 0, 0, 0, 0, 0, 0])
    wound_area_final, is_broken_sm, is_closed_sm, frame_broken, frame_closed = pp.run_full_postproc_sequence(wound_area, is_broken, is_closed)
    assert frame_broken == 5
    assert frame_closed is None
    assert np.all(np.isclose(is_broken_sm, is_broken))
    assert np.all(np.isclose(is_closed_sm, is_closed))
    assert wound_area_final.shape[0] == wound_area.shape[0]
    wound_area = np.asarray([100, 150, 125, 100, 100, 100, 100, 100])
    is_broken = np.asarray([0, 0, 0, 0, 0, 0, 0, 0])
    is_closed = np.asarray([0, 0, 0, 0, 0, 1, 1, 1])
    wound_area_final, is_broken_sm, is_closed_sm, frame_broken, frame_closed = pp.run_full_postproc_sequence(wound_area, is_broken, is_closed)
    assert frame_broken is None
    assert frame_closed == 5
    assert np.all(np.isclose(is_broken_sm, is_broken))
    assert np.all(np.isclose(is_closed_sm, is_closed))
    assert wound_area_final.shape[0] == wound_area.shape[0]
    assert wound_area_final[5] == 0
    assert wound_area_final[6] == 0
    assert wound_area_final[7] == 0


def test_get_postproc_results():
    folder_path = example_path("test_ph1_movie_mini_Anish")
    yaml_path = folder_path.joinpath("test_ph1_mini_movie.yaml").resolve()
    input_dict = ia._yml_to_dict(yml_path_file=yaml_path)
    path_dict = ia.input_info_to_output_paths(folder_path, input_dict)
    _, _ = ia.run_all(folder_path)
    output_path = path_dict["segment_ph1_path"]
    wound_area_final, is_broken_sm, is_closed_sm, frame_broken, frame_closed = pp.get_postproc_results(output_path)
    assert wound_area_final.shape[0] == 5
    assert is_broken_sm.shape[0] == 5
    assert is_closed_sm.shape[0] == 5
    assert frame_broken is None
    assert frame_closed is None
    output_path = path_dict["track_pillars_ph1_path"]
    pillar_disp_x, pillar_disp_y = pp.get_pillar_info(output_path)
    assert pillar_disp_x.shape[0] == 5
    assert pillar_disp_y.shape[0] == 5


def test_pos_to_disp():
    arr = np.random.random((10, 3))
    disp_arr = pp.pos_to_disp(arr)
    assert np.allclose(disp_arr[0, :], np.zeros((3, 1)))
    assert np.allclose(disp_arr[5, :], arr[5, :] - arr[0, :])


def test_get_drift_mat():
    mat, mat_inv = pp.get_drift_mat()
    mat_gt = np.asarray([[1., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 1.],
        [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 1., 1., 1., 0., 0.]])
    assert np.allclose(mat, mat_gt)
    assert np.allclose(mat_inv, np.linalg.inv(mat))


def test_get_defl_decoupled():
    defl_x = np.random.random((3, 4))
    defl_y = np.random.random((3, 4))
    _, mat_inv = pp.get_drift_mat()
    new_disp = pp.get_defl_decoupled(defl_x[1, :], defl_y[1, :], mat_inv)
    assert new_disp.shape == (10, 1)
    assert np.isclose(new_disp[0] + new_disp[8], defl_x[1, 0])[0]
    assert np.isclose(new_disp[1] + new_disp[8], defl_x[1, 1])[0]
    assert np.isclose(new_disp[2] + new_disp[8], defl_x[1, 2])[0]
    assert np.isclose(new_disp[3] + new_disp[8], defl_x[1, 3])[0]
    assert np.isclose(new_disp[4] + new_disp[9], defl_y[1, 0])[0]
    assert np.isclose(new_disp[5] + new_disp[9], defl_y[1, 1])[0]
    assert np.isclose(new_disp[6] + new_disp[9], defl_y[1, 2])[0]
    assert np.isclose(new_disp[7] + new_disp[9], defl_y[1, 3])[0]


def test_drift_correct_pillar_track_3p():
    folder_path = example_path("test_ph1_movie_mini_Anish")
    yaml_path = folder_path.joinpath("test_ph1_mini_movie.yaml").resolve()
    input_dict = ia._yml_to_dict(yml_path_file=yaml_path)
    path_dict = ia.input_info_to_output_paths(folder_path, input_dict)
    _, _ = ia.run_all(folder_path)
    output_path = path_dict["track_pillars_ph1_path"]
    pillar_disp_x, pillar_disp_y = pp.get_pillar_info(output_path)
    pillar_defl_x_gt = pp.pos_to_disp(pillar_disp_x)
    pillar_defl_y_gt = pp.pos_to_disp(pillar_disp_y)
    pillar_defl_x, pillar_defl_y, rigid_x, rigid_y = pp.drift_correct_pillar_track(output_path)
    assert np.allclose(pillar_defl_x_gt, pillar_defl_x)
    assert np.allclose(pillar_defl_y_gt, pillar_defl_y)
    assert np.allclose(np.zeros(rigid_x.shape), rigid_x)
    assert np.allclose(np.zeros(rigid_y.shape), rigid_y)


def test_drift_correct_pillar_track():
    folder_path = example_path("test_phi_movie_mini_Anish_tracking")
    yaml_path = folder_path.joinpath("test_ph1_mini_movie_with_tracking.yaml").resolve()
    input_dict = ia._yml_to_dict(yml_path_file=yaml_path)
    path_dict = ia.input_info_to_output_paths(folder_path, input_dict)
    _, _ = ia.run_all(folder_path)
    output_path = path_dict["track_pillars_ph1_path"]
    pillar_disp_x, pillar_disp_y = pp.get_pillar_info(output_path)
    pillar_defl_x_gt = pp.pos_to_disp(pillar_disp_x)
    pillar_defl_y_gt = pp.pos_to_disp(pillar_disp_y)
    pillar_defl_x, pillar_defl_y, rigid_x, rigid_y = pp.drift_correct_pillar_track(output_path)
    assert np.allclose(pillar_defl_x_gt, pillar_defl_x) is False
    assert np.allclose(pillar_defl_y_gt, pillar_defl_y) is False
    assert np.allclose(np.zeros(rigid_x.shape), rigid_x) is False
    assert np.allclose(np.zeros(rigid_y.shape), rigid_y) is False


def test_get_angle_and_distance_basic():
    """Test core functionality with a simple case"""
    angle, distance_sq = pp.get_angle_and_distance(3, 4, 0, 0)
    
    expected_angle = np.arctan2(4, 3)
    assert np.isclose(angle, expected_angle)
    assert distance_sq == 25


def test_order_points_clockwise_with_indices_basic_ordering():
    """Test points in all quadrants are ordered correctly."""
    points = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    ordered_points, ordered_indices = pp.order_points_clockwise_with_indices(points)

    assert np.allclose(ordered_points,np.array([[1,-1], [-1,-1], [-1,1], [1,1]]))
    assert ordered_indices == [3, 2, 1, 0]


def test_order_points_clockwise_with_indices_empty_input():
    """Test empty input returns empty lists."""
    points = []
    ordered_points, ordered_indices = pp.order_points_clockwise_with_indices(points)
    assert ordered_points == []
    assert ordered_indices == []

def test_order_points_clockwise_with_indices_collinear_points():
    """Test collinear points (same angle) are ordered by distance."""
    points = np.array([(1, 0), (2, 0), (3, 0)])  # All on x-axis
    ordered_points, ordered_indices = pp.order_points_clockwise_with_indices(points)
    
    # Expected order: closest to centroid first
    assert np.allclose(ordered_points,np.array([(1, 0), (2, 0), (3, 0)]))
    assert ordered_indices == [0, 1, 2]


def test_rearrange_pillars_indexing_basic_4_pillars():
    """Test basic functionality with 4 pillars in distinct quadrants"""
    # Create 4 simple pillar masks (small squares in each quadrant)
    img_h, img_w = 600, 1600
    pillar_masks = [
        np.zeros((img_h, img_w)),  # Will be placed in quadrant 0 (top-right)
        np.zeros((img_h, img_w)),  # quadrant 1 (top-left)
        np.zeros((img_h, img_w)),  # quadrant 2 (bottom-left)
        np.zeros((img_h, img_w))   # quadrant 3 (bottom-right)
    ]
    
    # Place centroids in each quadrant
    centroids = [(1200, 70), (20, 100), (10, 500), (1000, 450)]  # x,y coordinates
    for i, (x, y) in enumerate(centroids):
        pillar_masks[i][y-5:y+5, x-5:x+5] = 1  # 10x10 square centered at (x,y)
    
    x_locs = np.array([[1200, 20, 10, 1000], [1202, 22, 12, 1002]])
    y_locs = np.array([[70, 100, 500, 450], [72, 102, 502, 452]])
    
    new_masks, new_x, new_y = pp.rearrange_pillars_indexing(pillar_masks, x_locs, y_locs)
    
    # Verify location data is rearranged correctly
    expected_x_order = [10, 20, 1200, 1000]  # Based on original positions
    expected_y_order = [500, 100, 70, 450]

    assert np.allclose(new_x[0], expected_x_order)
    assert np.allclose(new_y[0], expected_y_order)


def test_compute_relative_pillars_dist_one_frame():
    """Test with a simple 2-object, 1-frame case."""
    x_locs = np.array([[0.0, 1.0]])  # Frame 1: p0 at x=0, p1 at x=1
    y_locs = np.array([[0.0, 0.0]])  # Frame 1: p0 at y=0, p1 at y=0

    distances, pair_names = pp.compute_relative_pillars_dist(x_locs, y_locs)

    # Expected distance: sqrt((1-0)^2 + (0-0)^2) = 1.0
    assert np.allclose(distances, np.array([[1.0]]))
    assert pair_names == ["p0-p1"]


def test_compute_relative_pillars_dist_multiple_frames():
    """Test with multiple frames."""
    x_locs = np.array([
        [0.0, 1.0],  # Frame 1
        [0.0, 2.0],  # Frame 2
    ])
    y_locs = np.array([
        [0.0, 0.0],  # Frame 1
        [0.0, 0.0],  # Frame 2
    ])

    distances, pair_names = pp.compute_relative_pillars_dist(x_locs, y_locs)

    # Expected distances:
    # Frame 1: sqrt(1^2 + 0^2) = 1.0
    # Frame 2: sqrt(2^2 + 0^2) = 2.0
    assert np.allclose(distances, np.array([[1.0], [2.0]]))
    assert pair_names == ["p0-p1"]


def test_compute_relative_pillars_dist_non_matching_shapes():
    """Test when x_locs and y_locs have different shapes."""
    x_locs = np.array([[0.0, 1.0]])
    y_locs = np.array([[0.0, 1.0, 2.0]])  # Different P

    with pytest.raises(ValueError):
        pp.compute_relative_pillars_dist(x_locs, y_locs)


def test_smooth_with_GPR_Matern_kernel_basic():
    """Test basic functionality with clean input data"""
    # Create simple sinusoidal test data
    num_frames = 100
    x = np.linspace(0, 10, num_frames)
    s = np.sin(x)  # Clean input signal
    
    smoothed = pp.smooth_with_GPR_Matern_kernel(s)
    
    # Basic output validation
    assert isinstance(smoothed, np.ndarray)
    assert smoothed.shape == (num_frames,)
    assert not np.any(np.isnan(smoothed))
    assert not np.any(np.isinf(smoothed))
    
    # Should be smoother than original (lower std dev)
    assert np.std(smoothed) <= np.std(s)


def test_smooth_with_GPR_Matern_kernel_with_nans():
    """Test handling of NaN values in input"""
    num_frames = 50
    s = np.random.randn(num_frames)
    
    # Add some NaN values
    s[10:15] = np.nan
    s[30:35] = np.nan
    
    smoothed = pp.smooth_with_GPR_Matern_kernel(s)
    
    # Should handle NaNs and return all finite values
    assert smoothed.shape == (num_frames,)
    assert not np.any(np.isnan(smoothed))
    assert not np.any(np.isinf(smoothed))
    
    # Check values around NaN regions are reasonable
    assert np.all(np.isfinite(smoothed[9:16]))
    assert np.all(np.isfinite(smoothed[29:36]))


def test_smooth_with_GPR_Matern_kernel_edge_cases():
    """Test edge cases - very short input and all NaN input"""
    # Very short input
    s_short = np.array([1.0, 2.0, 3.0])
    smoothed_short = pp.smooth_with_GPR_Matern_kernel(s_short)
    assert smoothed_short.shape == (3,)
    
    # All NaN input
    s_all_nan = np.full(10, np.nan)
    smoothed_nan = pp.smooth_with_GPR_Matern_kernel(s_all_nan)
    assert smoothed_nan.shape == (10,)
    assert np.all(np.isnan(smoothed_nan))  # Should either return NaNs or handle gracefully


def test_smooth_with_GPR_Matern_kernel_noisy_data():
    """Test with noisy input data"""
    np.random.seed(42)  # For reproducible results
    num_frames = 200
    x = np.linspace(0, 10, num_frames)
    clean_signal = np.sin(x)
    noisy_signal = clean_signal + 0.2 * np.random.randn(num_frames)
    
    smoothed = pp.smooth_with_GPR_Matern_kernel(noisy_signal)
    
    # Should be closer to clean signal than noisy input
    mse_noisy = np.mean((noisy_signal - clean_signal)**2)
    mse_smoothed = np.mean((smoothed - clean_signal)**2)
    assert mse_smoothed < mse_noisy


def test_smooth_relative_pillar_distances_with_GPR_basic():
    """Test basic functionality with clean input"""
    # Create test data: 100 frames, 3 pillar pairs
    num_frames = 100
    num_pairs = 3
    test_data = np.zeros((num_frames, num_pairs))
    
    # Create simple patterns for each pillar pair
    test_data[:, 0] = np.sin(np.linspace(0, 10, num_frames))  # Sinusoidal pattern
    test_data[:, 1] = np.linspace(0, 1, num_frames)          # Linear pattern
    test_data[:, 2] = np.random.randn(num_frames)             # Random pattern
    
    smoothed = pp.smooth_relative_pillar_distances_with_GPR(test_data)
    
    # Basic output validation
    assert isinstance(smoothed, np.ndarray)
    assert smoothed.shape == (num_frames, num_pairs)
    assert not np.any(np.isnan(smoothed))
    assert not np.any(np.isinf(smoothed))
    
    # Should maintain same general patterns
    assert np.allclose(np.diff(smoothed[:, 1]), np.diff(test_data[:, 1]), atol=0.1)  # Linear pattern preserved


def test_smooth_relative_pillar_distances_with_GPR_with_nans():
    """Test handling of NaN values in input"""
    num_frames = 50
    num_pairs = 2
    test_data = np.random.randn(num_frames, num_pairs)
    
    # Add NaN values to first pillar pair
    test_data[10:15, 0] = np.nan
    test_data[30:35, 0] = np.nan
    
    smoothed = pp.smooth_relative_pillar_distances_with_GPR(test_data)
    
    # Should handle NaNs and return all finite values
    assert smoothed.shape == (num_frames, num_pairs)
    assert not np.any(np.isnan(smoothed))
    
    # Check NaN regions are properly interpolated
    assert np.all(np.isfinite(smoothed[10:15, 0]))
    assert np.all(np.isfinite(smoothed[30:35, 0]))


def test_smooth_relative_pillar_distances_with_GPR_single_pair():
    """Test with single pillar pair"""
    num_frames = 30
    test_data = np.random.randn(num_frames, 1)  # Single pillar pair
    
    smoothed = pp.smooth_relative_pillar_distances_with_GPR(test_data)
    
    assert smoothed.shape == (num_frames, 1)
    assert not np.any(np.isnan(smoothed))


def test_compute_pillar_disps_between_frames():
    pillar_x_locs = pillar_y_locs= np.array([
        [0,0,0,0],
        [1,1,2,2],
        [2,2,4,4]
    ])
    known_x_disps = known_y_disps = np.array([
        [0,0,0,0],
        [1,1,2,2],
        [1,1,2,2]
    ])
    found_x_disps,found_y_disps = pp.compute_pillar_disps_between_frames(pillar_x_locs,pillar_y_locs)
    assert np.allclose(found_x_disps,known_x_disps)
    assert np.allclose(found_y_disps,known_y_disps)


def test_no_displacements_above_threshold():
    """Test when no displacements exceed the threshold"""
    px_disps = np.array([1, 2, 3, 4])
    py_disps = np.array([1, 2, 3, 4])
    disp_thresh = 10
    expected = (False, np.array([]))
    
    result = pp.check_large_pillar_disps(px_disps, py_disps, disp_thresh)
    assert result[0] == expected[0]
    np.testing.assert_array_equal(result[1], expected[1])


def test_single_x_displacement_above_threshold():
    """Test detection of single x displacement above threshold"""
    px_disps = np.array([11, 2, 3, 4])
    py_disps = np.array([1, 2, 3, 4])
    disp_thresh = 10
    expected = (True, np.array([0]))
    
    result = pp.check_large_pillar_disps(px_disps, py_disps, disp_thresh)
    assert result[0] == expected[0]
    np.testing.assert_array_equal(result[1], expected[1])


def test_single_y_displacement_above_threshold():
    """Test detection of single y displacement above threshold"""
    px_disps = np.array([1, 2, 3, 4])
    py_disps = np.array([1, 12, 3, 4])
    disp_thresh = 10
    expected = (True, np.array([1]))
    
    result = pp.check_large_pillar_disps(px_disps, py_disps, disp_thresh)
    assert result[0] == expected[0]
    np.testing.assert_array_equal(result[1], expected[1])


def test_multiple_displacements_both_axes():
    """Test multiple displacements in both axes with deduplication"""
    px_disps = np.array([11, 2, 13, 4])
    py_disps = np.array([1, 12, 3, 14])
    disp_thresh = 10
    expected = (True, np.array([0, 1, 2, 3]))
    
    result = pp.check_large_pillar_disps(px_disps, py_disps, disp_thresh)
    assert result[0] == expected[0]
    np.testing.assert_array_equal(result[1], expected[1])


def test_edge_case_at_threshold():
    """Test values exactly at threshold (should not be counted)"""
    px_disps = np.array([10, 2, 3, 4])
    py_disps = np.array([1, 2, 10, 4])
    disp_thresh = 10
    expected = (False, np.array([]))
    
    result = pp.check_large_pillar_disps(px_disps, py_disps, disp_thresh)
    assert result[0] == expected[0]
    np.testing.assert_array_equal(result[1], expected[1])


def test_empty_input_arrays():
    """Test handling of empty input arrays"""
    px_disps = np.array([])
    py_disps = np.array([])
    disp_thresh = 10
    expected = (False, np.array([]))
    
    result = pp.check_large_pillar_disps(px_disps, py_disps, disp_thresh)
    assert result[0] == expected[0]
    np.testing.assert_array_equal(result[1], expected[1])


def test_absolute_value_handling():
    """Test that negative displacements are properly handled with absolute values"""
    px_disps = np.array([-11, 2, 3, 4])
    py_disps = np.array([1, -12, 3, 4])
    disp_thresh = 10
    expected = (True, np.array([0, 1]))
    
    result = pp.check_large_pillar_disps(px_disps, py_disps, disp_thresh)
    assert result[0] == expected[0]
    np.testing.assert_array_equal(result[1], expected[1])


def test_duplicate_frame_indices():
    """Test that the same frame index from both axes is only reported once"""
    px_disps = np.array([11, 2, 3, 4])
    py_disps = np.array([11, 2, 3, 4])
    disp_thresh = 10
    expected = (True, np.array([0]))
    
    result = pp.check_large_pillar_disps(px_disps, py_disps, disp_thresh)
    assert result[0] == expected[0]
    np.testing.assert_array_equal(result[1], expected[1])


def test_check_potential_large_background_shift():
    """Test detection of potential large background shifts from pillar displacements"""
    # Setup test data - 3 frames, 2 pillars
    pillar_x_locs = np.array([
        [100, 200],  # Frame 0
        [105, 205],  # Frame 1 (small movement - no shift)
        [120, 220]   # Frame 2 (large movement - should trigger shift)
    ])
    pillar_y_locs = np.array([
        [50, 150],   # Frame 0
        [55, 155],   # Frame 1 (small movement - no shift)
        [50, 150]    # Frame 2 (y movement within threshold)
    ])
    
    # Test with default threshold (10 pixels)
    result_bool, result_frames = pp.check_potential_large_background_shift(
        pillar_x_locs, pillar_y_locs
    )
    print(result_frames)
    
    # Verify results
    assert result_bool is True  # Should detect shift in frame 2
    assert np.array_equal(result_frames, np.array([2]))  # Frame 1->2 displacement
    
    # Test with higher threshold (should not detect shift)
    result_bool, result_frames = pp.check_potential_large_background_shift(
        pillar_x_locs, pillar_y_locs, disp_thresh=25
    )
    assert result_bool is False
    assert len(result_frames) == 0


def test_compute_absolute_actual_pillar_disps():

    # no motion
    pillars_pos_x = np.array([[0, 1, 0, 1], 
                                [0, 1, 0, 1]])  # No movement
    pillars_pos_y = np.array([[0, 0, 1, 1],
                                [0, 0, 1, 1]])
    pillar_disps, avg_pillar_disps, actual_dx, actual_dy = pp.compute_absolute_actual_pillar_disps(
        pillars_pos_x, pillars_pos_y
    )
    np.testing.assert_array_almost_equal(pillar_disps[1], np.zeros(4))
    np.testing.assert_array_almost_equal(avg_pillar_disps, np.zeros(2))
    np.testing.assert_array_almost_equal(actual_dx[1], np.zeros(4))
    np.testing.assert_array_almost_equal(actual_dy[1], np.zeros(4))


    # rigid body motion
    pillars_pos_x = np.array([[0, 1, 0, 1], 
                                [0.1, 1.1, 0.1, 1.1]])  # 2 time points, 4 pillars
    pillars_pos_y = np.array([[0, 0, 1, 1],
                                [0.1, 0.1, 1.1, 1.1]])
    pillar_disps, avg_pillar_disps, actual_dx, actual_dy = pp.compute_absolute_actual_pillar_disps(
        pillars_pos_x, pillars_pos_y
    )
    zeros_arr = np.zeros_like(pillars_pos_x)
    assert np.allclose(pillar_disps, zeros_arr)
    assert np.allclose(actual_dx, zeros_arr)
    assert np.allclose(actual_dy, zeros_arr)
    assert np.allclose(avg_pillar_disps,np.zeros((2,)))


    # pillars have actual deflection
    # Arrange - pillars deform differently
    pillars_pos_x = np.array([[0, 1, 0, 1], 
                                [0.1, 1.2, 0.0, 1.0]])  # Different x movements
    pillars_pos_y = np.array([[0, 0, 1, 1],
                                [0.0, 0.1, 1.1, 0.9]])  # Different y movements
    pillar_disps, avg_pillar_disps, actual_dx, actual_dy = pp.compute_absolute_actual_pillar_disps(
        pillars_pos_x, pillars_pos_y
    )
    assert not np.allclose(pillar_disps[1], 0.0)
    assert not np.allclose(avg_pillar_disps[1], 0.0)
