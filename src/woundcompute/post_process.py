import numpy as np
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


def get_wound_area(output_path: Path) -> np.ndarray:
    file_name = "wound_area_vs_frame"
    file_path = output_path.joinpath(file_name + '.txt').resolve()
    wound_area = np.loadtxt(str(file_path))
    return wound_area


def get_is_broken(output_path: Path) -> np.ndarray:
    file_name = "is_broken_vs_frame"
    file_path = output_path.joinpath(file_name + '.txt').resolve()
    is_broken = np.loadtxt(str(file_path))
    return is_broken


def get_is_closed(output_path: Path) -> np.ndarray:
    file_name = "is_closed_vs_frame"
    file_path = output_path.joinpath(file_name + '.txt').resolve()
    is_closed = np.loadtxt(str(file_path))
    return is_closed


def smooth_with_GPR(s: np.ndarray) -> np.ndarray:
    num_frames = s.shape[0]
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e1)) + 1.0 * WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    xdata = np.arange(num_frames).reshape(-1, 1)
    xhat = xdata
    ydata = s.reshape(-1, 1)
    remove_indices = np.where(np.isnan(ydata.reshape(-1)))[0]
    xdata = np.delete(xdata, remove_indices, axis=0)
    ydata = np.delete(ydata, remove_indices, axis=0)
    model.fit(xdata, ydata)
    yhat = model.predict(xhat)
    return yhat.reshape(-1)


def is_broken_smooth(is_broken: np.ndarray) -> np.ndarray:
    # step 1: remove "blips"
    is_broken_no_blips = []
    is_broken_no_blips.append(0)
    for kk in range(1, is_broken.shape[0]-1):
        if is_broken[kk] == 1:
            if is_broken[kk-1] or is_broken[kk + 1]:
                is_broken_no_blips.append(1)
            else:
                is_broken_no_blips.append(0)
        else:
            is_broken_no_blips.append(0)
    is_broken_no_blips.append(is_broken[-1])
    is_broken_no_blips = np.asarray(is_broken_no_blips)
    # step 2: make irreversible
    is_broken_irreversible = []
    is_broken_irreversible.append(0)
    for kk in range(1, is_broken.shape[0]):
        if np.sum(is_broken_no_blips[0:kk]) > 0 or is_broken_no_blips[kk] == 1:
            is_broken_irreversible.append(1)
        else:
            is_broken_irreversible.append(0)
    return np.asarray(is_broken_irreversible)


def is_closed_smooth(is_closed: np.ndarray) -> np.ndarray:
    # step 1: remove "blips"
    is_closed_no_blips = []
    is_closed_no_blips.append(0)
    for kk in range(1, is_closed.shape[0]-1):
        if is_closed[kk] == 1:
            if is_closed[kk-1] or is_closed[kk + 1]:
                is_closed_no_blips.append(1)
            else:
                is_closed_no_blips.append(0)
        else:
            is_closed_no_blips.append(0)
    is_closed_no_blips.append(is_closed[-1])
    is_closed_no_blips = np.asarray(is_closed_no_blips)
    return is_closed_no_blips


def wound_area_with_is_closed(wound_area: np.ndarray, is_closed: np.ndarray) -> np.ndarray:
    wound_area_new = []
    for kk in range(0, wound_area.shape[0]):
        if is_closed[kk] == 0:
            wound_area_new.append(wound_area[kk])
        else:
            wound_area_new.append(0)
    return np.asarray(wound_area_new)

