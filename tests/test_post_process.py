import numpy as np
from pathlib import Path
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
