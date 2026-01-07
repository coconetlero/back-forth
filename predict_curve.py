import os
import time
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"

import cv2
import numpy as np
import torch
import yaml

import utils.Smoothing as smooth

from matplotlib import pyplot as plt
from scc_tree.SCC_Tree import SCC_Tree
from gru_model_evaluation import (
    CurveToWModel,
    CurveDataset,
    TrainingConfig,
    load_model,
)

def predict_w_from_curve(model, new_curve, device="cpu"):
    model.eval()

    # Convert to tensor
    if not isinstance(new_curve, torch.Tensor):
        points = torch.tensor(new_curve, dtype=torch.float32)
    else:
        points = new_curve.to(torch.float32)

    # Same normalization as during training
    points = CurveDataset._normalize_curve(points)  # (n, 2)

    length = torch.tensor([points.shape[0]], dtype=torch.long)  # (1,)
    padded_points = points.unsqueeze(0)  # (1, n, 2)

    padded_points = padded_points.to(device)
    length = length.to(device)

    with torch.no_grad():
        w_hat = model(padded_points, length)  # (1, 2)

    return w_hat.squeeze(0).cpu().numpy()  # (2,)



def open_tree_def(config_file_path):
    with open(config_file_path, 'r') as file:
        config_data = yaml.safe_load(file)

    sp = (config_data["start_position"]["y"], config_data["start_position"]["x"])
    image_path = config_data["base_folder"] + config_data["binary_image"]

    assert os.path.isfile(image_path), "The image {} doesn't exixt".format(config_data["binary_image"])
    o_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return [o_image, sp]



def plot_results(curve1, curve2):
    """
    Plot the results of the polynomial fitting
    """
    plt.figure(figsize=(12, 12))
    plt.plot(curve1[:, 0], curve1[:, 1], 'bo-', alpha=0.3, markersize=2, linewidth=1, label='Original')
    plt.plot(curve2[:, 0], curve2[:, 1], 'ro-', alpha=0.8, markersize=2, linewidth=1, label='Smoothed')


    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Curve Smoothing with Cubic Spline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()



def main():
    # Must match what you used in training
    config = TrainingConfig(
        hidden_dim=128,
        num_layers=1,
        model_dir="./models",
        model_name="curve_to_w_gru.pt",
    )
    device = config.device

    # Build model and load weights
    model = CurveToWModel(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    ).to(device)

    ok = load_model(model, config)
    if not ok:
        return

    # Load or construct a new curve: shape (n, 2)
    # Example dummy curve:
    # n = 50
    # new_curve = np.random.randn(n, 2).astype(np.float32)

    # w_pred = predict_w_from_curve(model, new_curve, device=device)
    # print("Predicted w:", w_pred)

    [image, tree_root] = open_tree_def("/Users/zianfanti/Trabajo/tree_representation/back-forth/config.yaml")
    # [image, tree_root] = open_tree_def("/Users/zianfanti/IIMAS/Tree_Representation/src/back-forth/config.yaml")
    scc_tree = SCC_Tree.create_from_image(image, tree_root)
    branches = scc_tree.get_pixelated_branches()
  
    for branch in branches:
        w_pred = predict_w_from_curve(model, branch, device=device)
        points_p = abs(w_pred[0])
        smooth_p = abs(w_pred[1])
        smooth_curve = smooth.smooth_with_regularization(branch, arclen_points=points_p, smoothing_factor=smooth_p)

        print("Predicted w:", w_pred)
        plot_results(branch, smooth_curve)
        


def find_optimal_parameters(curve_folder, image_folder):

    curves = sorted(os.listdir(curve_folder))
    images = sorted(os.listdir(image_folder))


     # Must match what you used in training
    config = TrainingConfig(
        hidden_dim=128,
        num_layers=1,
        model_dir="./models",
        model_name="curve_to_w_gru.pt",
    )
    device = config.device

    # Build model and load weights
    model = CurveToWModel(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    ).to(device)

    ok = load_model(model, config)
    if not ok:
        return
    
    for curve, image in zip(curves, images):
        curve_filename = os.path.join(curve_folder, curve)
        image_filename = os.path.join(image_folder, image)

        points = np.loadtxt(curve_filename, dtype=int)
        unique_rows, idx = np.unique(points, axis=0, return_index=True)
        pixelated_curve = unique_rows[np.argsort(idx)]

        w_pred = predict_w_from_curve(model, pixelated_curve, device=device)
        print("Predicted w:", w_pred)
        
        points_p = abs(w_pred[0])
        smooth_p = abs(w_pred[1])
        smooth_curve = smooth.smooth_with_regularization(pixelated_curve, arclen_points=points_p, smoothing_factor=smooth_p)

        x = smooth_curve[:, 0]
        y = smooth_curve[:, 1]

        img_bgr = cv2.imread(image_filename)
        if img_bgr is None:
            raise FileNotFoundError("Could not load image")

        # Convert to RGB for matplotlib
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(18, 12))
        plt.imshow(img_rgb)
        plt.plot(pixelated_curve[:, 0], pixelated_curve[:, 1], 'mo-', alpha=0.8, markersize=2, linewidth=1, label='Original')  # curve overlay        
        plt.plot(x, y, linewidth=2, color='lime', alpha=0.7) 
        plt.scatter(x, y, s=5, color='yellow', alpha=1.0)             
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
 

if __name__ == "__main__":

    start_time = time.perf_counter()

    # main()
    find_optimal_parameters('/Volumes/HOUSE MINI/IMAGENES/Fondus_Databases/RET-TORT/ROIs/arteries', 
                            '/Volumes/HOUSE MINI/IMAGENES/Fondus_Databases/RET-TORT/Reduced Arteries Iso')
    # find_optimal_parameters('/Volumes/HOUSE MINI/IMAGENES/Fondus_Databases/RET-TORT/ROIs/veins', 
    #                         '/Volumes/HOUSE MINI/IMAGENES/Fondus_Databases/RET-TORT/Reduced Veins Iso')



    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds") 
    
