import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage.morphology import binary_erosion

def normalized_surface_dice(pred_mask: np.ndarray, gt_mask: np.ndarray, tau: float = 1.0) -> float:
    """
    Calculates the Normalized Surface Dice (NSD) for a binary segmentation mask.
    NSD is a boundary-based metric that evaluates how well the predicted and ground truth
    surfaces align, tolerant to small deviations up to a threshold 'tau'.

    Args:
        pred_mask (np.ndarray): The binary prediction mask (boolean or 0/1).
        gt_mask (np.ndarray): The binary ground truth mask (boolean or 0/1).
        tau (float): The tolerance parameter in pixels. A distance <= tau is considered a match.

    Returns:
        float: The Normalized Surface Dice score, from 0.0 to 1.0.
    """
    # Ensure inputs are boolean arrays
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)

    # Handle empty masks case
    if not np.any(pred_mask) and not np.any(gt_mask):
        return 1.0  # Both are empty, perfect match
    if not np.any(pred_mask) or not np.any(gt_mask):
        return 0.0  # One is empty, the other is not, total mismatch

    # --- Extract surfaces (boundaries) ---
    # The boundary is the original mask minus its eroded version
    eroded_pred = binary_erosion(pred_mask)
    eroded_gt = binary_erosion(gt_mask)
    
    surface_pred = pred_mask ^ eroded_pred
    surface_gt = gt_mask ^ eroded_gt
    
    # Get coordinates of surface pixels
    coords_pred = np.argwhere(surface_pred)
    coords_gt = np.argwhere(surface_gt)
    
    # Handle case where a mask is solid (no boundary)
    if coords_pred.shape[0] == 0 or coords_gt.shape[0] == 0:
        return 0.0

    # --- Calculate distances between surfaces ---
    # cdist computes the distance between each pair of the two collections of inputs
    dist_matrix = cdist(coords_pred, coords_gt)

    # Find the minimum distance from each pred point to any gt point
    min_dist_pred_to_gt = np.min(dist_matrix, axis=1)
    
    # Find the minimum distance from each gt point to any pred point
    min_dist_gt_to_pred = np.min(dist_matrix, axis=0)

    # --- Calculate NSD score ---
    # Count how many points are within the tolerance tau
    num_correct_pred = np.sum(min_dist_pred_to_gt <= tau)
    num_correct_gt = np.sum(min_dist_gt_to_pred <= tau)
    
    # Total number of surface points
    total_points_pred = coords_pred.shape[0]
    total_points_gt = coords_gt.shape[0]
    
    # NSD formula
    nsd = (num_correct_pred + num_correct_gt) / (total_points_pred + total_points_gt)
    
    return nsd