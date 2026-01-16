import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import re
import os
import time

import utils.Smoothing as smooth
import utils.load_and_write as lw
import ImageToSCC as imscc
import Morphology_Measurements_Single_Curve as measure

from typing import Callable, Tuple, List
from dataclasses import dataclass

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


import numpy as np
from typing import Callable, Tuple, List
from dataclasses import dataclass


# ============================
# Bounds helper
# ============================
@dataclass
class Bounds:
    lows: np.ndarray   # shape (d,)
    highs: np.ndarray  # shape (d,)


# ============================
# Sampling + GP helpers
# ============================
def sample_random_w(bounds: Bounds, n_samples: int) -> np.ndarray:
    """
    Uniform random sampling for w in given bounds.
    Returns: array of shape (n_samples, d)
    """
    return bounds.lows + (bounds.highs - bounds.lows) * np.random.rand(n_samples, bounds.lows.size)


def fit_gp(X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    """
    Fit a Gaussian Process to data (X, y).
    X: (n_samples, d), y: (n_samples,)
    """
    d = X.shape[1]

    kernel = ConstantKernel(1.0, (0.1, 10.0)) * Matern(length_scale=np.ones(d), nu=2.5) \
             + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-9, 1e-1))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=0,
    )
    gp.fit(X, y)
    return gp


def expected_improvement(
    X: np.ndarray,
    model: GaussianProcessRegressor,
    y_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """
    EI for a MAXIMIZATION problem at candidate points X.
    X: (n_points, d)
    Returns: (n_points,)
    """
    mu, sigma = model.predict(X, return_std=True)
    mu = mu.reshape(-1)
    sigma = sigma.reshape(-1)

    sigma = np.maximum(sigma, 1e-9)  # avoid divide-by-zero

    improvement = mu - y_best - xi
    Z = improvement / sigma

    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma <= 0.0] = 0.0
    return ei


def argmax_acquisition_random(
    acquisition_fn: Callable[[np.ndarray], np.ndarray],
    bounds: Bounds,
    n_candidates: int = 1000,
) -> np.ndarray:
    """
    Maximize acquisition via random search.
    Returns: best w (d,)
    """
    X_cand = sample_random_w(bounds, n_candidates)
    acq_vals = acquisition_fn(X_cand)
    idx = np.argmax(acq_vals)
    return X_cand[idx]


# ============================
# Main BO loop for one input x
# ============================
def optimize_w_for_input(
    x: np.ndarray,
    metrics_fn: Callable[[np.ndarray, np.ndarray], Tuple[float, float]],
    a: float,
    b: float,
    bounds: Bounds,
    n_init: int = 5,
    n_iter: int = 25,
    n_candidates: int = 1000,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Minimize J(x, w) = a*u(x, w) + b*v(x, w) over w using Bayesian optimization.

    Args:
        x: input for this run, shape (2, n) for example (curve vector)
        metrics_fn: function (x, w) -> (u, v)  (two real-valued metrics)
        a, b: weights in J = a*u + b*v
        bounds: Bounds for w (dim = 2 here)
        n_init: number of random initial samples
        n_iter: number of BO iterations
        n_candidates: random candidates for acquisition maximization
        verbose: whether to print progress

    Returns:
        best_w: (2,) optimal parameters found
        best_J: float, minimal J(x, w)
        best_u: float, u(x, best_w) u -> amount of points in the curve (%)
        best_v: float, v(x, best_w) v -> smoothing value
    """
    # We will MAXIMIZE -J, since our EI is for maximization.

    # 1) Initial random samples
    Xw = sample_random_w(bounds, n_init)  # (n_init, d)
    J_list: List[float] = []
    u_list: List[float] = []
    v_list: List[float] = []

    for i in range(n_init):
        w_i = Xw[i]
        Td_i, D_i = metrics_fn(x, w_i)
        t1 = a * Td_i
        t2 = b * D_i
        J_i = a * Td_i + b * D_i
        J_list.append(J_i)
        u_list.append(Td_i)
        v_list.append(D_i)

        with open("/Users/zianfanti/Trabajo/tree_representation/back-forth/train/Td_i_-_D_i.csv", 'a') as f:            
            f.write('{:.6f}, {:.6f} \n'.format(Td_i, D_i))

        if verbose:
            print(f"[INIT] {i+1}/{n_init} | u={Td_i:.4f}, v={D_i:.4f}, J={J_i:.4f}, w={w_i}")

    J_arr = np.array(J_list)
    y = -J_arr  # for maximization

    # 2) BO iterations
    for t in range(n_iter):
        gp = fit_gp(Xw, y)
        y_best = np.max(y)

        def acquisition(Xcand: np.ndarray) -> np.ndarray:
            return expected_improvement(Xcand, gp, y_best=y_best, xi=0.01)

        # Propose next w
        w_next = argmax_acquisition_random(acquisition, bounds, n_candidates)

        # Evaluate true metrics at (x, w_next)
        u_next, v_next = metrics_fn(x, w_next)
        J_next = (a * u_next) + (b * v_next)
        y_next = -J_next

        # Append
        Xw = np.vstack([Xw, w_next])
        y = np.append(y, y_next)
        J_arr = np.append(J_arr, J_next)
        u_list.append(u_next)
        v_list.append(v_next)

        if verbose:
            print(f"[ITER {t+1}/{n_iter}] u={u_next:.4f}, v={v_next:.4f}, "
                  f"J={J_next:.4f}, w={w_next}, best_J={np.min(J_arr):.4f}")

    # 3) Best (minimum J)
    best_idx = int(np.argmin(J_arr))
    best_w = Xw[best_idx]
    best_J = float(J_arr[best_idx])
    best_u = float(u_list[best_idx])
    best_v = float(v_list[best_idx])

    if verbose:
        print("\n=== Optimization finished ===")
        # print(f"Best J = {best_J:.4f} (min)")
        # print(f"Best u = {best_u:.4f}, v = {best_v:.4f}")
        # print(f"Best w = {best_w}")

    return best_w, best_J, best_u, best_v



# ============================
# find random x from the set
# ============================
def get_random_curve(path, image_folder, des_file) -> np.ndarray:
    with open(os.path.join(path, des_file), 'r', encoding='utf-8') as f: 
        lines = f.readlines()
        line = random.choice(lines).strip()

        match = re.search(r'(\S+)\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)', line)               
        if match:
            fname = match.group(1)
            match2 = re.search(r'_(\d+)_X(\d+)', fname)
            if match2:
                file_num = int(match2.group(1))
                scale = float(match2.group(2)) / 10.0

                # load original curve (spline generated)
                name, _ = os.path.splitext(fname)                
                original_curve = np.loadtxt(os.path.join(path, "points", name + ".txt"), delimiter=',')
                original_curve *= scale
                
                # load start position
                x = float(match.group(2))
                y = float(match.group(3))
                sp = (int(y), int(x))


                # get curve from image
                o_image = cv2.imread(os.path.join(path, image_folder, fname), cv2.IMREAD_GRAYSCALE)
                treepath = imscc.build_tree(o_image, sp)                                           

                k = 2
                branch = []
                curve_elem = treepath[k]
                while type(curve_elem) is tuple:
                    branch.append(curve_elem)
                    curve_elem = treepath[k]
                    k += 1
                
                bx = np.array([point[0] for point in branch])
                by = np.array([point[1] for point in branch])

                pixel_curve = np.column_stack([bx, by])
                return [fname, pixel_curve, original_curve, scale]







# ============================
# Example usage / template
# ============================
if __name__ == "__main__":
   

    def smoothing_metrics(scale_original_curve: dict, w: np.ndarray) -> Tuple[float, float]:
        # x: (2, n)
        # w: (2,)    

        original_curve = scale_original_curve["original"]
        pixel_curve = scale_original_curve["test"]
        scale = scale_original_curve["scale"]

        points_v, smooth_v = w        
        smoothed_curve = smooth.smooth_with_regularization(pixel_curve, arclen_points=points_v, smoothing_factor=smooth_v)

        # plot_results(original_curve, smoothed_curve)

        [To, _] = measure.SCC(original_curve)
        [T, _] = measure.SCC(smoothed_curve)
        
        Td = abs(To - T)
        D = (smooth.average_min_distance(smoothed_curve, original_curve) / scale) / smoothed_curve.shape[0]
        return float(Td), float(D)

    

    # [random_curve, scale] = get_random_curve('/Users/zianfanti/IIMAS/images_databases/curves', "images", "coordinates_curves.txt")
    # [curve_name, random_curve, original_curve, scale] = get_random_curve('/Volumes/HOUSE MINI/IMAGENES/curves', "images", "coordinates_curves.txt")

    def obtain_best_params_for_all(path, image_folder, des_file):        
        names = []
        torts = []        
        dists = []
        params = []
        with open(os.path.join(path, des_file), 'r', encoding='utf-8') as f:        
            for idx, line in enumerate(f):            
                match1 = re.search(r'(\S+)\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)', line)               
                if match1:
                    fname = match1.group(1)                    
                    x = int(match1.group(2))
                    y = int(match1.group(3))

                match2 = re.search(r'_(\d+)_X(\d+)', fname)
                if match2:
                    scale = float(match2.group(2)) / 10.
                    
                # load original curve (spline generated)
                name, _ = os.path.splitext(fname)                
                original_curve = np.loadtxt(os.path.join(path, "points", name + ".txt"), delimiter=',')
                original_curve *= scale
                
                # start position
                sp = (y, x)
                pixel_curve = lw.load_pixelated_curve_from_image(os.path.join(path, image_folder, fname), sp)

                # lw.plot_two_curves(original_curve, pixel_curve, label1='Original', label2='Pixelated')

                a = 0.5      # Tortuosity distance ponderation
                b = 0.5      # Curves distance ponderation
                interations = 50

                # Bounds for w = [w1, w2]
                lows = np.array([0.1, 0.01], dtype=float)
                highs = np.array([1.0, 0.2], dtype=float) 
                bounds = Bounds(lows=lows, highs=highs)                

                scale_random_curve = {
                    "test": pixel_curve,
                    "original": original_curve,
                    "scale": scale 
                }

                best_w, best_J, best_u, best_v = optimize_w_for_input(
                    x=scale_random_curve,
                    metrics_fn=smoothing_metrics,
                    a=a,
                    b=b,
                    bounds=bounds,
                    n_init=5,
                    n_iter=interations,
                    n_candidates=500,
                    verbose=False,
                )

                names.append(fname)
                torts.append(best_u)
                dists.append(best_v)
                params.append(best_w)
                
                # with open("/Users/zianfanti/Trabajo/tree_representation/back-forth/train/test_output.csv", 'a') as f:
                #     f.write('{:.4f}, {:.4f}, {:.6f}, {:.6f} \n'.format(best_u, best_v, best_w[0], best_w[1]))
                    
                

        # for idx in range(len(names)):
        #     # print('{:<4}, {}, {}, {:.4f}, {:.4f}'.format(idx, names[idx], np.array2string(np.array(params[idx]), precision=4), torts[idx], dists[idx]))
        #     print('{:<4}, {:<20}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(idx, names[idx], params[idx][0], params[idx][1], torts[idx], dists[idx]))

        output_filename = ("/Users/zianfanti/Trabajo/tree_representation/back-forth/train/50iter_2500samples_t.csv")
        with open(output_filename, 'w') as f:
            for idx in range(len(names)):
                f.write('{:<4}, {:<20}, {:.6f}, {:.6f}, {:.6f}, {:.6f} \n'.format(idx, names[idx], params[idx][0], params[idx][1], torts[idx], dists[idx]))




    def obtain_best_params_for_all_2(path, pixel_curve_folder, target_curves_folder, des_file):        
        names = []
        torts = []        
        dists = []
        params = []

        [pixel_curves, pixel_fnames] = lw.load_pixelated_curves_from_txt_files(os.path.join(path, pixel_curve_folder))
        [target_curves, target_fnames] = lw.load_float_curves_from_txt_files(os.path.join(path, target_curves_folder))

        for (px_c, tg_c, fname) in zip(pixel_curves, target_curves, pixel_fnames):
            match = re.search(r'_(\d+)_X(\d+)', fname)
            if match:
                scale = float(match.group(2)) / 10.

            a = 0.5      # Tortuosity distance ponderation
            b = 0.5      # Curves distance ponderation
            interations = 50

            # Bounds for w = [w1 (amount of points), w2 (smoothing value)]
            lows = np.array([0.1, 0.01], dtype=float)
            highs = np.array([1.0, 0.2], dtype=float) 
            bounds = Bounds(lows=lows, highs=highs) 

            scale_random_curve = {
                "test": px_c,
                "original": tg_c,
                "scale": scale 
            }

            best_w, best_J, best_u, best_v = optimize_w_for_input(
                x=scale_random_curve,
                metrics_fn=smoothing_metrics,
                a=a,
                b=b,
                bounds=bounds,
                n_init=5,
                n_iter=interations,
                n_candidates=500,
                verbose=False,
            )

            names.append(fname)
            torts.append(best_u)
            dists.append(best_v)
            params.append(best_w)


        output_filename = ("/Users/zianfanti/Trabajo/tree_representation/back-forth/train/50iter_2500samples_t.csv")
        with open(output_filename, 'w') as f:
            for idx in range(len(names)):
                f.write('{:<4}, {:<20}, {:.6f}, {:.6f}, {:.6f}, {:.6f} \n'.format(idx, names[idx], params[idx][0], params[idx][1], torts[idx], dists[idx]))



    start_time = time.perf_counter()

    # obtain_best_params_for_all('/Users/zianfanti/IIMAS/images_databases/curves', "images", "coordinates_curves.txt")

    obtain_best_params_for_all_2('/Users/zianfanti/IIMAS/images_databases/curves_500_5', "pixel_curves", "target_scaled", "coordinates_curves.txt")
    
    
    # obtain_best_params_for_all('/Volumes/HOUSE MINI/IMAGENES/curves_500_5', "images", "coordinates_curves.txt")

    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds") 


    # # Define weights a, b
    # a = 0.5
    # b = 0.5

    # # Bounds for w = [w1, w2]
    # lows = np.array([0.1, 0.01], dtype=float)
    # highs = np.array([1.0, 0.2], dtype=float) 
    # bounds = Bounds(lows=lows, highs=highs)

    # scale_random_curve = {
    #     "test": random_curve,
    #     "original": original_curve,
    #     "scale": scale 
    # }

    # best_w, best_J, best_u, best_v = optimize_w_for_input(
    #     x=scale_random_curve,
    #     metrics_fn=smoothing_metrics,
    #     a=a,
    #     b=b,
    #     bounds=bounds,
    #     n_init=5,
    #     n_iter=25,
    #     n_candidates=500,
    #     verbose=False,
    # )

    # print('\nCurve name:', curve_name)
    # print("\nFinal best_w:", best_w)
    # print("Final best_J:", best_J)
    # print("Final best Tort diff:", best_u)
    # print("Final best Distance:", best_v)






