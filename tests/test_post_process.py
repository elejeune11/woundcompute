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
