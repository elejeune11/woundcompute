import numpy as np
from typing import List,Tuple
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from woundcompute import segmentation as seg


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
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e2)) + 1.0 * WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-14, 1e1))
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
    is_closed_no_blips.append(is_closed[0])
    is_closed_no_blips.append(is_closed[1])
    for kk in range(2, is_closed.shape[0]-2):
        if is_closed[kk] == 1:
            if is_closed[kk - 2] and is_closed[kk - 1] or is_closed[kk + 1] and is_closed[kk + 2]:
                is_closed_no_blips.append(1)
            elif is_closed[kk - 1] and is_closed[kk + 1]:
                is_closed_no_blips.append(1)
            else:
                is_closed_no_blips.append(0)
        else:
            is_closed_no_blips.append(0)
    is_closed_no_blips.append(is_closed[-2])
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


def run_full_postproc_sequence(wound_area: np.ndarray, is_broken: np.ndarray, is_closed: np.ndarray) -> np.ndarray:
    is_broken_sm = is_broken_smooth(is_broken)
    is_closed_sm = is_closed_smooth(is_closed)
    wound_area_new = wound_area_with_is_closed(wound_area, is_closed_sm)
    wound_area_new_smooth = smooth_with_GPR(wound_area_new)
    wound_area_final = wound_area_with_is_closed(wound_area_new_smooth, is_closed_sm)
    if np.sum(is_broken_sm) > 0:
        frame_broken = np.min(np.argwhere(is_broken_sm == 1))
    else:
        frame_broken = None
    if np.sum(is_closed_sm) > 0:
        frame_closed = np.min(np.argwhere(is_closed_sm == 1))
    else:
        frame_closed = None
    # return:
    # wound_area_smooth, is_broken_smooth, is_closed_smooth
    # frame_broken (first frame when broken) -- None if it doesn't break
    # frame closed (first frame when closed) -- None if it doesn't close
    return wound_area_final, is_broken_sm, is_closed_sm, frame_broken, frame_closed


def get_postproc_results(output_path: Path):
    wound_area = get_wound_area(output_path)
    is_broken = get_is_broken(output_path)
    is_closed = get_is_closed(output_path)
    wound_area_final, is_broken_sm, is_closed_sm, frame_broken, frame_closed = run_full_postproc_sequence(wound_area, is_broken, is_closed)
    return wound_area_final, is_broken_sm, is_closed_sm, frame_broken, frame_closed


def get_pillar_info(output_path: Path) -> np.ndarray:
    file_name = "pillar_tracker_x"
    file_path = output_path.joinpath(file_name + '.txt').resolve()
    pillar_disp_x = np.loadtxt(str(file_path))
    file_name = "pillar_tracker_y"
    file_path = output_path.joinpath(file_name + '.txt').resolve()
    pillar_disp_y = np.loadtxt(str(file_path))
    return pillar_disp_x, pillar_disp_y


def pos_to_disp(pos_arr: np.array):
    disp_arr = np.zeros(pos_arr.shape)
    for kk in range(0, pos_arr.shape[0]):
        disp_arr[kk, :] = pos_arr[kk, :] - pos_arr[0, :]
    return disp_arr


def get_drift_mat():
    mat = np.zeros((10, 10))
    for kk in range(0, 8):
        mat[kk, kk] = 1
    for kk in range(0, 4):
        mat[kk, 8] = 1
    for kk in range(4, 8):
        mat[kk, 9] = 1
    mat[8, :] = np.asarray([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    mat[9, :] = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    mat_inv = np.linalg.inv(mat)
    return mat, mat_inv


def get_defl_decoupled(defl_x: np.ndarray, defl_y: np.ndarray, mat_inv: np.ndarray) -> np.ndarray:
    vec = np.zeros((10, 1))
    vec[0:4, 0] = defl_x[:]
    vec[4:8, 0] = defl_y[:]
    new_disp = np.dot(mat_inv, vec)
    return new_disp


def drift_correct_pillar_track(output_path: Path):
    pillar_disp_x, pillar_disp_y = get_pillar_info(output_path)
    pillar_defl_x = pos_to_disp(pillar_disp_x)
    pillar_defl_y = pos_to_disp(pillar_disp_y)
    num_frames = pillar_disp_x.shape[0]
    num_pillars = pillar_disp_x.shape[1]
    rigid_x = np.zeros((pillar_disp_x.shape[0], 1))
    rigid_y = np.zeros((pillar_disp_y.shape[0], 1))
    if num_pillars == 4:
        pillar_defl_x_update = np.zeros(pillar_defl_x.shape)
        pillar_defl_y_update = np.zeros(pillar_defl_y.shape)
        _, mat_inv = get_drift_mat()
        for jj in range(0, num_frames):
            new_disp = get_defl_decoupled(pillar_defl_x[jj, :], pillar_defl_y[jj, :], mat_inv)
            pillar_defl_x_update[jj, :] = new_disp[0:4, 0]
            pillar_defl_y_update[jj, :] = new_disp[4:8, 0]
            rigid_x[jj, :] = new_disp[8, 0]
            rigid_y[jj, :] = new_disp[9, 0]
        return pillar_defl_x_update, pillar_defl_y_update, rigid_x, rigid_y
    else:
        return pillar_defl_x, pillar_defl_y, rigid_x, rigid_y


def get_angle_and_distance(x, y, cx, cy):
    """
    Calculate the angle (in radians) and squared Euclidean distance from a point (x, y) 
    to a reference center point (cx, cy).

    :param x (float or int): X-coordinate of the target point.
    :param y (float or int): Y-coordinate of the target point.
    :param cx (float or int): X-coordinate of the reference center point.
    :param cy (float or int): Y-coordinate of the reference center point.

    :returns:
        tuple[float, float]: A tuple containing:
            - angle (float): Angle in radians from the positive x-axis (centered at (cx, cy)), 
              in the range [-π, π] (computed via `np.arctan2`).
            - distance (float): Squared Euclidean distance (dx² + dy²) to avoid sqrt overhead.
    """
    dx = x - cx
    dy = y - cy
    angle = np.arctan2(dy, dx)  # [-pi, pi]
    distance = dx**2 + dy**2     # Squared distance (avoid sqrt)
    return angle, distance


def order_points_clockwise_with_indices(points):
    if len(points)==0:
        return [], []
    
    # Compute centroid (origin for angle calculation)
    n = len(points)
    cx = sum(x for x, y in points) / n
    cy = sum(y for x, y in points) / n
    
    # Process points: (original index, x, y, angle, distance)
    processed = []
    for idx, (x, y) in enumerate(points):
        angle, dist = get_angle_and_distance(x, y, cx, cy)
        processed.append((idx, x, y, angle, dist))
    
    # Shift angles to [0, 2pi) for consistent sorting
    shifted_processed = []
    for idx, x, y, angle, dist in processed:
        shifted_angle = angle if angle >= 0 else angle + 2 * np.pi
        shifted_processed.append((idx, x, y, shifted_angle, dist))
    
    # Sort by angle, then by distance
    shifted_processed.sort(key=lambda p: (-p[3], p[4]))
    
    # Extract ordered points and original indices
    ordered_points = [(x, y) for _, x, y, _, _ in shifted_processed]
    ordered_indices = [idx for idx, _, _, _, _ in shifted_processed]
    
    return ordered_points, ordered_indices


def rearrange_pillars_indexing(pillar_masks:List,x_locs:np.ndarray,y_locs:np.ndarray):
    """
    Rearrange the pillars' location.
    First pillar is top right, second pillar top left,
    third pillar bottom right, fourth pillar bottom right.
    We could update this for multiple pillars by computing angle to center.

    :param pillar_masks: A list of binary masks depicting the pillars.
    :param x_locs: x-location of pillars. N by P array, where N is frame number, P is number of pillars.
    :param y_locs: y-location of pillars. N by P array, where N is frame number, P is number of pillars.

    :return: List of pillar masks. Rearranged x and y locations of pillars.
    """

    img_h,img_w = pillar_masks[0].shape
    num_pillars = len(pillar_masks)
    new_pillar_masks = np.zeros((num_pillars,img_h,img_w))
    new_x_locs = np.zeros_like(x_locs)
    new_y_locs = np.zeros_like(y_locs)

    pillar_centroids = np.zeros((num_pillars,2))
    for mask_ind,mask in enumerate(pillar_masks):
        region_props = seg.get_region_props(mask)
        region_prop = seg.get_largest_regions(region_props,1)
        pm_center = region_prop[0].centroid
        pillar_centroids[mask_ind,:] = pm_center
    
    _,sorted_ind = order_points_clockwise_with_indices(pillar_centroids)
    
    for new_ind,og_ind in enumerate(sorted_ind):
        new_pillar_masks[new_ind,:,:] = pillar_masks[og_ind]
        new_x_locs[:,new_ind] = x_locs[:,og_ind]
        new_y_locs[:,new_ind] = y_locs[:,og_ind]

    return new_pillar_masks,new_x_locs,new_y_locs


def compute_relative_pillars_dist(pil_x_locs:np.ndarray,pil_y_locs:np.ndarray):
    """
    Compute the relative distance (Euclidean) between all pillars.

    """

    if pil_x_locs.shape[1] != pil_y_locs.shape[1]:
        raise ValueError("The x-location array and y-location array must have the same dimension.")
    
    num_frames,num_pillars = pil_x_locs.shape

    # Generate all unique pairs of objects
    pairs = []
    for i in range(num_pillars):
        for j in range(i + 1, num_pillars):
            pairs.append((i, j))

    # Compute distances for each frame and pair
    distances = np.zeros((num_frames, len(pairs)))
    for n in range(num_frames):
        for r, (i, j) in enumerate(pairs):
            dx = pil_x_locs[n, i] - pil_x_locs[n, j]
            dy = pil_y_locs[n, i] - pil_y_locs[n, j]
            distances[n, r] = np.sqrt(dx**2 + dy**2)

    # Generate pair names
    pair_names = np.array([f"p{i}-p{j}" for (i, j) in pairs])

    return distances,pair_names


def smooth_with_GPR_Matern_kernel(s: np.ndarray) -> np.ndarray:
    num_frames = s.shape[0]
    kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e2), nu=2.5) + 1.0 * WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-14, 1e1))
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    xdata = np.arange(num_frames).reshape(-1, 1)
    xhat = xdata
    ydata = s.reshape(-1, 1)
    remove_indices = np.where(np.isnan(ydata.reshape(-1)))[0]
    xdata = np.delete(xdata, remove_indices, axis=0)
    ydata = np.delete(ydata, remove_indices, axis=0)
    if len(xdata) == 0 or len(ydata) == 0:
        return np.full(s.shape, np.nan)  # Return NaN array if no data to fit
    model.fit(xdata, ydata)
    yhat = model.predict(xhat)
    return yhat.reshape(-1)


def smooth_relative_pillar_distances_with_GPR(relative_pillar_distances):
    """
    Given an array of relative distances between pillars,
    smooth the distances with GPR (Matern kernel).

    :param relative_pillar_distances: array of relative pillar distances
    :return GPR_relative_distances: GPR smoothed relative distances
    """
    num_frames,num_pairs = relative_pillar_distances.shape
    GPR_relative_distances = np.zeros((num_frames,num_pairs))
    for pair_ind in range(num_pairs):
        cur_rel_dist = relative_pillar_distances[:,pair_ind]
        smoothed_rel_dist = smooth_with_GPR_Matern_kernel(cur_rel_dist)
        GPR_relative_distances[:,pair_ind] = smoothed_rel_dist
    return GPR_relative_distances


def compute_pillar_disps_between_frames(pillar_x_locs:np.ndarray,pillar_y_locs:np.ndarray)->np.ndarray:
    """
    Computes the displacement of pillars between consecutive frames.
    
    For each frame (except the first), this function calculates how much each pillar has moved
    from its position in the previous frame. The first frame will have zero displacement
    since there's no previous frame to compare with.
    
    :param pillar_x_locs: np.ndarray
        A 2D array of shape (num_frames, num_pillars) containing x-coordinates of pillars
        for each frame. Each row represents a frame, each column represents a pillar.
    :param pillar_y_locs: np.ndarray
        A 2D array of shape (num_frames, num_pillars) containing y-coordinates of pillars
        for each frame. Must have same dimensions as pillar_x_locs.
        
    :return tuple[np.ndarray, np.ndarray]:
        A tuple containing two 2D numpy arrays:
        - px_disp_changing_ref: Array of x-axis displacements between frames
        - py_disp_changing_ref: Array of y-axis displacements between frames
        Both arrays have same shape as input arrays, with first row being zeros.
    """
    px_disp_between_frames = np.zeros_like(pillar_x_locs)
    py_disp_between_frames = np.zeros_like(pillar_y_locs)

    num_frames,num_pillars = pillar_x_locs.shape

    for frame_ind in range(num_frames):
        if frame_ind == num_frames-1:
            break
        px_disp_between_frames[frame_ind+1,:] = pillar_x_locs[frame_ind+1,:] - pillar_x_locs[frame_ind,:]
        py_disp_between_frames[frame_ind+1,:] = pillar_y_locs[frame_ind+1,:] - pillar_y_locs[frame_ind,:]
    
    return px_disp_between_frames,py_disp_between_frames


def check_large_pillar_disps(px_disps:np.ndarray,py_disps:np.ndarray,disp_thresh:int=10)->Tuple[bool,np.ndarray]:
    """
    This function takes the absolute x displacements and y displacements of pillars, and determine potential large background
    shift according to a threshold.
    """
    px_disps = np.abs(px_disps)
    py_disps = np.abs(py_disps)
    large_disp_frame_ind_x = np.argwhere(px_disps>disp_thresh)[:,0]
    large_disp_frame_ind_y = np.argwhere(py_disps>disp_thresh)[:,0]
    large_disp_frame_ind = np.unique(np.concatenate((large_disp_frame_ind_x,large_disp_frame_ind_y)))

    if large_disp_frame_ind.size == 0:
        return False,large_disp_frame_ind
    else:
        return True,large_disp_frame_ind
    

def check_potential_large_background_shift(pillar_x_locs:np.ndarray,pillar_y_locs:np.ndarray,disp_thresh:int=10)->Tuple[bool,np.ndarray]:
    """
    Detects potential large background shifts by analyzing pillar displacements between consecutive frames.
    
    This function first computes the displacement of each pillar between consecutive frames, then identifies
    any frames where displacements exceed the specified threshold, indicating a potential large background shift.
    
    :param pillar_x_locs: np.ndarray
        A 2D array of shape (num_frames, num_pillars) containing x-coordinates of pillars for each frame.
        Each row represents a frame, each column represents a pillar.
    :param pillar_y_locs: np.ndarray
        A 2D array of shape (num_frames, num_pillars) containing y-coordinates of pillars for each frame.
        Must have same dimensions as pillar_x_locs.
    :param disp_thresh: int, optional
        The displacement threshold (in pixels) for considering a movement as a potential large shift.
        Default is 10 pixels.
        
    :return: Tuple[bool, np.ndarray]
        A tuple containing:
        - bool: True if any frame contains pillar displacements exceeding the threshold (potential shift detected),
                False otherwise
        - np.ndarray: Array of frame indices where large displacements were detected (empty if none found)
        
    :raises ValueError:
        If pillar_x_locs and pillar_y_locs have different shapes
    """
    px_disp_between_frames,py_disp_between_frames = compute_pillar_disps_between_frames(pillar_x_locs,pillar_y_locs)
    is_potential_shift,large_disp_frame_ind = check_large_pillar_disps(px_disp_between_frames,py_disp_between_frames,disp_thresh)
    return is_potential_shift,large_disp_frame_ind


def compute_pillar_disps(
    pillars_pos_x:np.ndarray,
    pillars_pos_y:np.ndarray,
    ):
    """
    Compute the absolute pillar displacements and average pillar displacements,
    after removing background drift.
    """
    
    dx_measured = pillars_pos_x - pillars_pos_x[0,:]
    dy_measured = pillars_pos_y - pillars_pos_y[0,:]

    dx_drift = np.sum(dx_measured,axis=1)/4
    dy_drift = np.sum(dy_measured,axis=1)/4
    dx_drift = dx_drift.reshape(-1,1)
    dy_drift = dy_drift.reshape(-1,1)

    actual_dx = dx_measured - dx_drift
    actual_dy = dy_measured - dy_drift
    
    abs_actual_pillar_disps = np.sqrt(actual_dx**2 + actual_dy**2)
    avg_actual_disps = np.sum(abs_actual_pillar_disps,axis=1)/4

    return abs_actual_pillar_disps,avg_actual_disps,actual_dx,actual_dy
