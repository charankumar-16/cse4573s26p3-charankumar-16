'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import torch

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    detection_results: List[List[float]] = []

    ##### YOUR IMPLEMENTATION STARTS HERE #####
    if img.dim() == 3 and img.shape[0] == 3:
      img_hwc = img.permute(1, 2, 0)
    else:
        img_hwc = img
 
    if img_hwc.dtype != torch.uint8:
        img_hwc = img_hwc.to(torch.uint8)
 
    img_np = img_hwc.numpy()
 
 
    locations = face_recognition.face_locations(img_np, model='hog')
 
    for (top, right, bottom, left) in locations:
        x = float(left)
        y = float(top)
        width = float(right - left)
        height = float(bottom - top)
        detection_results.append([x, y, width, height]) 

    return detection_results



def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    cluster_results: List[List[str]] = [[] for _ in range(K)] # Please make sure your output follows this data format.
        
    ##### YOUR IMPLEMENTATION STARTS HERE #####
    image_names = []
    face_vectors = []  
 
    for image_name, image_tensor in imgs.items():
        if image_tensor.dim() == 3 and image_tensor.shape[2] == 3:
            image_hwc = image_tensor 
        elif image_tensor.dim() == 3 and image_tensor.shape[0] == 3:
            image_hwc = image_tensor.permute(1, 2, 0)
        else:
            image_hwc = image_tensor
 
        if image_hwc.dtype != torch.uint8:
            image_hwc = image_hwc.to(torch.uint8)
 
        image_np = image_hwc.numpy()
 
        face_locations = face_recognition.face_locations(image_np, model='hog')
 
        if len(face_locations) == 0:
            image_height, image_width = image_np.shape[:2]
            face_locations = [(0, image_width, image_height, 0)]
 
        encoding_list = face_recognition.face_encodings(image_np, known_face_locations=face_locations)
 
        if len(encoding_list) > 0:
            face_vector_tensor = torch.tensor(encoding_list[0], dtype=torch.float32)
        else:
            face_vector_tensor = torch.zeros(128, dtype=torch.float32)
 
        image_names.append(image_name)
        face_vectors.append(face_vector_tensor)
 
    if len(face_vectors) == 0:
        return cluster_results
 
    feature_matrix = torch.stack(face_vectors, dim=0)  # (N, 128)
 
    cluster_labels = _kmeans(feature_matrix, K, max_iters=300, n_init=10)
 
    cluster_results = [[] for _ in range(K)]
    for image_index, cluster_id in enumerate(cluster_labels):
        cluster_results[cluster_id].append(image_names[image_index])
 
    return cluster_results
 
 
def _kmeans(data: torch.Tensor, K: int, max_iters: int = 300, n_init: int = 10) -> List[int]:
    """
    K-Means clustering implemented from scratch using PyTorch.
 
    Args:
        data: torch.Tensor of shape (N, D)
        K: number of clusters
        max_iters: maximum iterations per run
        n_init: number of random restarts; best result (lowest inertia) is kept
 
    Returns:
        labels: List[int] of length N, cluster assignment for each sample
    """
    num_samples, num_features = data.shape
 
    best_labels = None
    best_inertia = float('inf')
 
    for _ in range(n_init):
        random_order = torch.randperm(num_samples)
        cluster_centers = data[random_order[0]].unsqueeze(0)  
 
        for center_index in range(1, K):
            distance_diff = data.unsqueeze(1) - cluster_centers.unsqueeze(0) 
            squared_distances = (distance_diff ** 2).sum(dim=2)                 
            min_squared_distances, _ = squared_distances.min(dim=1)                  
 
            selection_probs = min_squared_distances / (min_squared_distances.sum() + 1e-10)
            selection_probs_list = selection_probs.tolist()
            cumulative_prob = 0.0
            random_value = float(torch.rand(1).item())
            selected_index = num_samples - 1
            for sample_index, sample_prob in enumerate(selection_probs_list):
                cumulative_prob += sample_prob
                if random_value <= cumulative_prob:
                    selected_index = sample_index
                    break
            cluster_centers = torch.cat([cluster_centers, data[selected_index].unsqueeze(0)], dim=0)
 
        sample_labels = torch.zeros(num_samples, dtype=torch.long)
 
        for _ in range(max_iters):
            distance_diff = data.unsqueeze(1) - cluster_centers.unsqueeze(0) 
            squared_distances = (distance_diff ** 2).sum(dim=2)                   
            updated_labels = squared_distances.argmin(dim=1)                  
 
            if torch.equal(updated_labels, sample_labels):
                break
            sample_labels = updated_labels
 
            updated_centers = torch.zeros(K, num_features, dtype=data.dtype)
 
            for cluster_index in range(K):
                cluster_mask = (sample_labels == cluster_index)
                if cluster_mask.sum() > 0:
                    updated_centers[cluster_index] = data[cluster_mask].mean(dim=0)
                else:
                    updated_centers[cluster_index] = data[torch.randint(num_samples, (1,)).item()]
 
            cluster_centers = updated_centers
 
        distance_diff = data.unsqueeze(1) - cluster_centers.unsqueeze(0)  
        squared_distances = (distance_diff ** 2).sum(dim=2)                  
        assigned_squared_distances = squared_distances[torch.arange(num_samples), sample_labels]
        inertia = assigned_squared_distances.sum().item()
 
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = sample_labels.tolist()
 
    return best_labels
 

'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)
