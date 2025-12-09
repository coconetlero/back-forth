import math
import numpy as np



from typing import Tuple, Optional
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree




def smooth_Savitzky_Golay(x, y, size, window_length=11, polyorder=3):
    # Calculate cumulative arc length
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate(([0], np.cumsum(ds)))
    
    # Resample to uniform arc length
    s_uniform = np.linspace(0, s[-1], size)
    x_uniform = np.interp(s_uniform, s, x)
    y_uniform = np.interp(s_uniform, s, y)
    
    # Apply S-G filter
    x_smooth = savgol_filter(x_uniform, window_length, polyorder)
    y_smooth = savgol_filter(y_uniform, window_length, polyorder)
    
    smooth_curve = np.column_stack([x_smooth, y_smooth])
    return smooth_curve




def smooth_with_univariate_spline(pixel_curve, smoothing_factor=1.1, num_points=400):
    """
    Smooth x and y separately as functions of arc length
    """

    # size = int(len(pixel_curve[:, 0]) * 0.25)
    size = pixel_curve.shape[0] if num_points == 0 else num_points

    # Calculate arc length parameter
    dx = np.diff(pixel_curve[:, 0])
    dy = np.diff(pixel_curve[:, 1])
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate(([0], np.cumsum(ds)))  # Arc length parameter
    
    x = pixel_curve[:, 0]
    y = pixel_curve[:, 1]
    
    # Smooth x and y separately as functions of arc length
    spline_x = UnivariateSpline(s, x, s=smoothing_factor * len(x))
    spline_y = UnivariateSpline(s, y, s=smoothing_factor * len(y))
    
    # Generate smooth arc length parameter
    s_smooth = np.linspace(0, s.max(), size)
    
    # Get smoothed x and y
    x_smooth = spline_x(s_smooth)
    y_smooth = spline_y(s_smooth)
    
    return np.column_stack([x_smooth, y_smooth])



def smooth_with_regularization(pixel_curve, arclen_points=0.25, smoothing_factor=0.05):
    """
    Smooth both x and y with curvature regularization
    """
    px = pixel_curve[:, 0]
    py = pixel_curve[:, 1]
    num_points = round(len(px) * arclen_points) 
    arclen_curve = arclen_parametrization(px, py, num_points)
    
    assert arclen_curve.shape[0] > 3, "The curve size need to have at least 4 points"
    smoothed_curve = smooth_with_univariate_spline(arclen_curve, smoothing_factor=smoothing_factor, num_points=num_points)   
    smoothed_curve[0] = pixel_curve[0]
    smoothed_curve[-1] = pixel_curve[-1]

    return smoothed_curve



def gaussian_smooth(points, closed=False, ds_resample=0.5, sigma=1.0, beta=0.01):
    """
    points: list/array of (x,y) pixels ordered along an 8-connected curve
    closed: wrap ends (True for closed contours)
    ds_resample: arc-length spacing for uniform samples (in pixel units)
    sigma: Gaussian sigma (in samples) for light smoothing of coordinates
    Returns: s, x, y, dx, dy, ddx, ddy, kappa, theta
    """
    L = arc_length(points)
    sigma_arc = beta * math.sqrt(L)
    sigma = sigma_arc if sigma == 0 else sigma

    print('{:.4f}'.format(sigma))

    P = np.asarray(points, dtype=float)  # shape (N, 2) as (x,y)
    if closed:
        # avoid duplicate last=first
        if np.allclose(P[0], P[-1]):
            P = P[:-1]
        P_ext = np.vstack([P, P[0]])  # close
    else:
        P_ext = P

    # 1) cumulative arc-length with 4/8 step lengths 1 and sqrt(2)
    seg = np.diff(P_ext, axis=0)
    step = np.hypot(seg[:,0], seg[:,1])
    s = np.concatenate([[0], np.cumsum(step)])
    x = P_ext[:,0]; y = P_ext[:,1]

    # 2) build uniform arc-length samples
    L = s[-1]
    if closed:
        # drop last point to avoid duplication at L
        s = s[:-1]; x = x[:-1]; y = y[:-1]; L = s[-1]
    su = np.arange(0, L, ds_resample)
    # linear interpolation of coordinates vs arc-length
    xu = np.interp(su, s, x)
    yu = np.interp(su, s, y)

    # 3) optional denoising (reduces staircase artifacts)
    if sigma and sigma > 0:
        # circular padding for closed curves
        mode = 'wrap' if closed else 'nearest'
        xu = gaussian_filter1d(xu, sigma=sigma, mode=mode)
        yu = gaussian_filter1d(yu, sigma=sigma, mode=mode)

    return np.column_stack([xu, yu])



def arc_length(points, closed=False):
    """
    points: array-like of (x, y) integer pixel coords ordered along the curve
    closed: if True, join last -> first (ignore duplicate if already closed)
    """
    P = np.asarray(points, dtype=float)
    if closed:
        if np.allclose(P[0], P[-1]):
            P = P[:-1]
        P = np.vstack([P, P[0]])  # close the loop
    d = np.diff(P, axis=0)
    step = np.hypot(d[:,0], d[:,1])          # 1 for 4-neigh, √2 for diagonals
    step = step[step > 0]                    # ignore accidental repeats

    return float(step.sum())



def arclen_parametrization(px, py, num_points):
    if num_points > 1:
        # equally spaced in arclength
        N = np.transpose(np.linspace(0, 1, num_points))

        # how many points will be uniformly interpolated?
        nt = N.size

        # number of points on the curve
        n = px.size
        pxy = np.column_stack([px, py])
        pt = np.zeros((nt, 2))

        # Compute the arclength of each segment.
        chordlen = (np.sum(np.diff(pxy, axis=0) ** 2, axis=1)) ** (1 / 2)
        # Normalize the arclengths to a unit total
        chordlen = chordlen / np.sum(chordlen)
        # cumulative arclength
        cumarc = np.append(0, np.cumsum(chordlen))

        tbins = np.digitize(N, cumarc)  # bin index in which each N is in

        # catch any problems at the ends
        tbins[np.where(np.bitwise_or(tbins <= 0, (N <= 0)))] = 1
        tbins[np.where(np.bitwise_or(tbins >= n, (N >= 1)))] = n - 1

        s = np.divide((N - cumarc[tbins - 1]), chordlen[tbins - 1])
        pt = pxy[tbins - 1, :] + np.multiply((pxy[tbins, :] - pxy[tbins - 1, :]), (np.vstack([s] * 2)).T)

        pt[0] = np.array([px[0], py[0]])
        pt[-1] = np.array([px[-1], py[-1]])
        return pt
    else:
        return np.array([(px[0], py[0]),(px[-1], py[-1])])



def smooth_with_arc_length(x, y, size):
    # Calculate cumulative arc length
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    length = np.sum(ds)
    ratio = abs(1 - (length / len(dx))) if (length / len(dx)) > 1 else (length / len(dx)) 
    ratio1 = len(dx) / length 
    # print("Curve ratio: {}".format(ratio))  

    size = round(len(dx) * (ratio)) 
    s = np.concatenate(([0], np.cumsum(ds)))
    
    # Resample to uniform arc length
    s_uniform = np.linspace(0, s[-1], size)
    x_uniform = np.interp(s_uniform, s, x)
    y_uniform = np.interp(s_uniform, s, y)
    
    smooth_curve = np.column_stack([x_uniform, y_uniform])
    return smooth_curve
    


def uniform_resample(points, n, closed=False, duplicate_endpoint=False):
    """
    Resample an 8-connected pixel curve to exactly n points, uniformly by arc-length.

    points : array-like of (x, y) in visiting order (ints or floats)
    n      : target number of points (>= 2 for open curves, >= 3 recommended for closed)
    closed : if True, treat curve as a loop
    duplicate_endpoint :
        - For closed=False (open): ignored (endpoints always included).
        - For closed=True: if True, return n+1 points with last == first (explicit closure).
                           if False (default), return exactly n distinct points around the loop.

    Returns
    -------
    (n, 2) or (n+1, 2) array of resampled points (floats).
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("points must be shape (N,2)")

    # Drop duplicate last=first if closed
    if closed and np.allclose(P[0], P[-1]):
        P = P[:-1]

    # Remove accidental consecutive duplicates (zero-length steps)
    keep = np.ones(len(P), dtype=bool)
    keep[1:] = np.any(np.diff(P, axis=0) != 0, axis=1)
    P = P[keep]
    if len(P) < 2:
        return P[:1].copy()  # degenerate curve

    # Build cumulative arc-length
    if closed:
        P_ext = np.vstack([P, P[0]])
    else:
        P_ext = P
    d = np.diff(P_ext, axis=0)
    seg = np.hypot(d[:,0], d[:,1])
    s = np.concatenate([[0.0], np.cumsum(seg)])

    if closed:
        # Interp support over [0, L] with end mapped to start
        L = s[-1]
        if L == 0:
            return P[:1].copy()
        s_interp = s  # already 0..L
        x_interp = np.append(P[:,0], P[0,0])
        y_interp = np.append(P[:,1], P[0,1])

        if duplicate_endpoint:
            # n distinct points + repeated start at end
            su = np.linspace(0.0, L, n+1)
        else:
            # exactly n distinct points around the loop
            su = np.linspace(0.0, L, n+1)[:-1]
    else:
        # open curve: s already 0..L with coordinates P
        L = s[-1]
        if L == 0:
            # If all points identical but n>1 requested, just tile the same point
            return np.tile(P[:1], (n,1))
        s_interp = s
        x_interp = P_ext[:,0]
        y_interp = P_ext[:,1]
        su = np.linspace(0.0, L, n)

    xu = np.interp(su, s_interp, x_interp)
    yu = np.interp(su, s_interp, y_interp)
    out = np.column_stack([xu, yu])

    return out


def average_min_distance(curve1, curve2):
    """
    Computes the average distance from each point in curve1
    to the nearest point in curve2.

    Parameters:
        curve1 (np.ndarray): Array of shape (n, 2) with (x, y) points.
        curve2 (np.ndarray): Array of shape (m, 2) with (x, y) points.

    Returns:
        float: The average of the minimum distances.
    """
    # Build a KD-tree for efficient nearest-neighbor search
    tree = cKDTree(curve2)
    
    # Query the nearest neighbor in curve2 for each point in curve1
    distances, _ = tree.query(curve1)
    
    # Compute and return the mean distance
    return np.sum(distances)