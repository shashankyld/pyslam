import numpy as np
from scipy.spatial import Delaunay, KDTree
import cv2
import networkx as nx
import torch
import open3d as o3d

def get_connected_components(graph):
  """
  Divides a graph into its connected components.

  Args:
    graph: A networkx graph object.

  Returns:
    A list of sets, where each set contains the nodes in a connected component.
  """
  return list(nx.connected_components(graph))

def get_dynamic_components(graph):
    connected_components = get_connected_components(graph)
    # Remove the components with the highest number of nodes and return the rest
    if len(connected_components) > 1:
        # Find the component with the highest number of nodes
        max_nodes = max(connected_components, key=len)
        # Remove the component with the highest number of nodes
        connected_components.remove(max_nodes)
    return connected_components

def delaunay_dynamic_visualization(slam):
    ''' 
    Visualizes the dynamic objects detected using Delaunay triangulation on the frame's image.
    '''
    frame = slam.tracking.f_cur
    img = frame.img.copy()
    points_2d = frame.kpsu.copy()  
    
    connected_components = get_dynamic_components(filter_delaunay_edges_by_3d_distance(slam))

    for component in connected_components:
        if len(component) < 3:  # Adjust threshold as needed
            continue

        for node in component:
            node = tuple(int(coord) for coord in node)
            cv2.circle(img, node, 3, (0, 0, 255), -1)

    return img



'''
# Example usage:
graph = nx.Graph()
graph.add_edges_from([(1, 2), (2, 3), (4, 5)])  # Example graph with two connected components

connected_components = get_connected_components(graph)
print(connected_components)  # Output: [{1, 2, 3}, {4, 5}]
'''

def delaunay_visualization(slam):

    ''' 
    Takes in the current time stamp and slam object and runs delaunay triangulation on the current frame
    
    '''
    # Getting access to the current frame properties after being populated by the SLAM system
    cur_frame = slam.tracking.f_cur 
    points_2d = cur_frame.kpsu.copy().astype(int)
    tri = Delaunay(points_2d)
    # print("Delaunay Triangulation Done", tri.simplices)
    img = cur_frame.img.copy() 
    for simplex in tri.simplices:
        cv2.line(img, tuple(points_2d[simplex[0]]), tuple(points_2d[simplex[1]]), (0, 255, 0), 1)
        cv2.line(img, tuple(points_2d[simplex[1]]), tuple(points_2d[simplex[2]]), (0, 255, 0), 1)
        cv2.line(img, tuple(points_2d[simplex[2]]), tuple(points_2d[simplex[0]]), (0, 255, 0), 1)
    return img

def convert_delauany_to_networkx(tri):
    ''' 
    Takes in the delaunay triangulation and converts it to a networkx graph object
    
    '''
    graph = nx.Graph()
    for simplex in tri.simplices:
        graph.add_edges_from([(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])])
    return graph

def delaunay_triangulation(slam):
    tri = Delaunay(slam.tracking.f_cur.kpsu.copy().astype(int))
    graph = convert_delauany_to_networkx(tri)
    return graph

'''
def draw_simplicies_on_image(slam):
    curr_frame = slam.tracking.f_cur
    img = curr_frame.img.copy()
    tri = Delaunay(curr_frame.kpsu.copy().astype(int))
    for simplex in tri.simplices:
        cv2.line(img, tuple(curr_frame.kpsu[simplex[0]].astype(int)), tuple(curr_frame.kpsu[simplex[1]].astype(int)), (0, 255, 0), 1)
        cv2.line(img, tuple(curr_frame.kpsu[simplex[1]].astype(int)), tuple(curr_frame.kpsu[simplex[2]].astype(int)), (0, 255, 0), 1)
        cv2.line(img, tuple(curr_frame.kpsu[simplex[2]].astype(int)), tuple(curr_frame.kpsu[simplex[0]].astype(int)), (0, 255, 0), 1)
    return img
'''

def convert_frame_to_kdtree(slam, id=-1):
    ''' 
    Input : SLAM object, id of the frame to be converted to a KDTree
    Output : Dictionary with the keypoints as the keys and the neighbors, desc and id as the values
    '''


    if id == -1:
        curr_frame = slam.tracking.f_cur
        img = curr_frame.img.copy()
        kpsu = curr_frame.kpsu.copy().astype(int)
        kp_desc = curr_frame.des.copy()
    elif id == -2:
        curr_frame = slam.map.get_frame(-2)
        img = curr_frame.img.copy()
        kpsu = curr_frame.kpsu.copy().astype(int)
        kp_desc = curr_frame.des.copy()
    kdtree = KDTree(kpsu) # Creating a KDTree from the keypoints
    dict = {}
    for i in range(len(kpsu)):
        dict[tuple(kpsu[i])] = {}
        dict[tuple(kpsu[i])]['desc'] = kp_desc[i]
        dict[tuple(kpsu[i])]['id'] = i
        dict[tuple(kpsu[i])]['neighbors'] = []
    
    simplicies = Delaunay(kpsu)
    for simplex in simplicies.simplices:
        # For each edge, without duplicates - add to each kp in the dictionary - its neighbors desc 
        '''
        ### LIKE THIS ####
        # dict = {"(x1,y1)": {"desc" : "desc1", "id" : 1, "neighbors" : ["(kp2, desc2)", "(kp3, desc3)"]}}
        '''
        for i in range(3):
            if tuple(kpsu[simplex[i]]) not in dict:
                # Will only happen if the point is not in the dictionary - which should not happen
                print("Error: Point not in dictionary")
                dict[tuple(kpsu[simplex[i]])] = {}
                dict[tuple(kpsu[simplex[i]])]['desc'] = kp_desc[simplex[i]]
                dict[tuple(kpsu[simplex[i]])]['id'] = simplex[i]
                dict[tuple(kpsu[simplex[i]])]['neighbors'] = []
            for j in range(3):
                # Add the neighbors to the dictionary
                if i != j: # Avoid adding the same point as a neighbor
                    # Check if the neighbor is already in the list
                    # print(dict[tuple(kpsu[simplex[i]])]['neighbors'])
                    
                    # if (kpsu[simplex[j]], kp_desc[simplex[j]]) not in dict[tuple(kpsu[simplex[i]])]['neighbors']:
                    #     dict[tuple(kpsu[simplex[i]])]['neighbors'].append((tuple(kpsu[simplex[j]]), kp_desc[simplex[j]]))
                    
                    # dict[tuple(kpsu[simplex[i]])]['neighbors'].append((tuple(kpsu[simplex[j]]), kp_desc[simplex[j]]))

                    if (tuple(kpsu[simplex[j]]), tuple(kp_desc[simplex[j]])) not in dict[tuple(kpsu[simplex[i]])]['neighbors']:
                        dict[tuple(kpsu[simplex[i]])]['neighbors'].append((tuple(kpsu[simplex[j]]), tuple(kp_desc[simplex[j]])))
    return dict

def convert_frame_to_kdtree_masked(slam, idxs_cur=[], id=-1):
    ''' 
    Input : SLAM object, id of the frame to be converted to a KDTree
    Output : Dictionary with the keypoints as the keys and the neighbors, desc and id as the values
    '''
    # TODO - TRY TO FIX THE DUPLICATES PROBLEM


    if id == -1:
        curr_frame = slam.tracking.f_cur
        img = curr_frame.img.copy()
        kpsu = curr_frame.kpsu.copy().astype(int)
        kp_desc = curr_frame.des.copy()
    elif id == -2:
        curr_frame = slam.map.get_frame(-2)
        img = curr_frame.img.copy()
        kpsu = curr_frame.kpsu.copy().astype(int)
        kp_desc = curr_frame.des.copy()

    # if len(idxs_cur) != len(np.unique(idxs_cur)):
    #     print("Warning: Duplicates found in idxs_cur")
    #     values, counts = np.unique(idxs_cur, return_counts=True)
    #     print("Duplicates:", values[counts > 1])


    print("Length of kpsu: ", len(kpsu)) 
    
    print("Length of idxs_cur: ", len(idxs_cur))
    # if len(idxs_cur) != len(np.unique(idxs_cur)):
    #     print("Warning: Duplicates found in idxs_cur")
    #     values, counts = np.unique(idxs_cur, return_counts=True)
    #     print("Duplicates:", len(values[counts > 1]), values[counts > 1])

    if len(idxs_cur) !=0:
        kpsu = kpsu[idxs_cur]
        kp_desc = kp_desc[idxs_cur]
        # now len(kpsu) == len(kp_desc) == len(idxs_cur) 


    print("Length of kpsu after masking: ", len(kpsu))

    # Remove duplicates from the keypoints and descriptors
    kpsu, indices = np.unique(kpsu, axis=0, return_index=True)
    kp_desc = kp_desc[indices]

    print("Length of kpsu after removing duplicates: ", len(kpsu))


    # Add early exit for insufficient points
    if len(kpsu) < 3:
        print("Warning: Not enough keypoints for Delaunay (%d)" % len(kpsu))
        return {}

    # kdtree = KDTree(kpsu) # Creating a KDTree from the keypoints
    dict = {}
    for i in range(len(kpsu)):
        dict[tuple(kpsu[i])] = {}
        dict[tuple(kpsu[i])]['desc'] = kp_desc[i]
        dict[tuple(kpsu[i])]['id'] = i
        dict[tuple(kpsu[i])]['neighbors'] = []
    
    simplicies = Delaunay(kpsu)
    for simplex in simplicies.simplices:
        # For each edge, without duplicates - add to each kp in the dictionary - its neighbors desc 
        '''
        ### LIKE THIS ####
        # dict = {"(x1,y1)": {"desc" : "desc1", "id" : 1, "neighbors" : ["(kp2, desc2)", "(kp3, desc3)"]}}
        '''
        for i in range(3):
            if tuple(kpsu[simplex[i]]) not in dict:
                # Will only happen if the point is not in the dictionary - which should not happen
                print("Error: Point not in dictionary")
                dict[tuple(kpsu[simplex[i]])] = {}
                dict[tuple(kpsu[simplex[i]])]['desc'] = kp_desc[simplex[i]]
                dict[tuple(kpsu[simplex[i]])]['id'] = simplex[i]
                dict[tuple(kpsu[simplex[i]])]['neighbors'] = []
            for j in range(3):
                # Add the neighbors to the dictionary
                if i != j: # Avoid adding the same point as a neighbor
                    # Check if the neighbor is already in the list
                    # print(dict[tuple(kpsu[simplex[i]])]['neighbors'])
                    
                    # if (kpsu[simplex[j]], kp_desc[simplex[j]]) not in dict[tuple(kpsu[simplex[i]])]['neighbors']:
                    #     dict[tuple(kpsu[simplex[i]])]['neighbors'].append((tuple(kpsu[simplex[j]]), kp_desc[simplex[j]]))
                    
                    # dict[tuple(kpsu[simplex[i]])]['neighbors'].append((tuple(kpsu[simplex[j]]), kp_desc[simplex[j]]))

                    if (tuple(kpsu[simplex[j]]), tuple(kp_desc[simplex[j]])) not in dict[tuple(kpsu[simplex[i]])]['neighbors']:
                        dict[tuple(kpsu[simplex[i]])]['neighbors'].append((tuple(kpsu[simplex[j]]), tuple(kp_desc[simplex[j]])))
    return dict

# def convert_frame_to_reference_dict(slam, id=-1):
#     ''' 
#     Input : SLAM object, id of the frame to be converted to a KDTree
#     Output : Dictionary with the keypoints as the keys and the neighbors, desc and id as the values, but for only the  keypoints that are common with the previous frame - given by idxs_cur 
#     '''


#     if id == -1:
#         curr_frame = slam.tracking.f_cur
#         img = curr_frame.img.copy()
#         kpsu = curr_frame.kpsu.copy().astype(int)
#         kp_desc = curr_frame.des.copy()
#     elif id == -2:
#         curr_frame = slam.map.get_frame(-2)
#         img = curr_frame.img.copy()
#         kpsu = curr_frame.kpsu.copy().astype(int)
#         kp_desc = curr_frame.des.copy()
#     kdtree = KDTree(kpsu) # Creating a KDTree from the keypoints
#     dict = {}
#     for i in range(len(kpsu)):
#         dict[tuple(kpsu[i])] = {}
#         dict[tuple(kpsu[i])]['desc'] = kp_desc[i]
#         dict[tuple(kpsu[i])]['id'] = i
#         dict[tuple(kpsu[i])]['neighbors'] = []
    
#     simplicies = Delaunay(kpsu)
#     for simplex in simplicies.simplices:
#         # For each edge, without duplicates - add to each kp in the dictionary - its neighbors desc 
#         '''
#         ### LIKE THIS ####
#         # dict = {"(x1,y1)": {"desc" : "desc1", "id" : 1, "neighbors" : ["(kp2, desc2)", "(kp3, desc3)"]}}
#         '''
#         for i in range(3):
#             if tuple(kpsu[simplex[i]]) not in dict:
#                 # Will only happen if the point is not in the dictionary - which should not happen
#                 print("Error: Point not in dictionary")
#                 dict[tuple(kpsu[simplex[i]])] = {}
#                 dict[tuple(kpsu[simplex[i]])]['desc'] = kp_desc[simplex[i]]
#                 dict[tuple(kpsu[simplex[i]])]['id'] = simplex[i]
#                 dict[tuple(kpsu[simplex[i]])]['neighbors'] = []
#             for j in range(3):
#                 # Add the neighbors to the dictionary
#                 if i != j: # Avoid adding the same point as a neighbor
#                     # Check if the neighbor is already in the list
#                     # print(dict[tuple(kpsu[simplex[i]])]['neighbors'])
                    
#                     # if (kpsu[simplex[j]], kp_desc[simplex[j]]) not in dict[tuple(kpsu[simplex[i]])]['neighbors']:
#                     #     dict[tuple(kpsu[simplex[i]])]['neighbors'].append((tuple(kpsu[simplex[j]]), kp_desc[simplex[j]]))
                    
#                     # dict[tuple(kpsu[simplex[i]])]['neighbors'].append((tuple(kpsu[simplex[j]]), kp_desc[simplex[j]]))

#                     if (tuple(kpsu[simplex[j]]), tuple(kp_desc[simplex[j]])) not in dict[tuple(kpsu[simplex[i]])]['neighbors']:
#                         dict[tuple(kpsu[simplex[i]])]['neighbors'].append((tuple(kpsu[simplex[j]]), tuple(kp_desc[simplex[j]])))
#     return dict


def convert_frame_to_reference_dict(slam, idxs_cur, id=-1):
    ''' 
    Input : SLAM object, idxs_cur (indices of keypoints common with previous frame), id of the frame to be converted to a KDTree
    Output : Dictionary with only the common keypoints as the keys and the neighbors, desc, and id as the values
    '''
    if id == -1:
        curr_frame = slam.tracking.f_cur
    elif id == -2:
        curr_frame = slam.map.get_frame(-2)
    else:
        raise ValueError("Invalid frame ID")
    
    img = curr_frame.img.copy()
    kpsu = curr_frame.kpsu.copy().astype(int)
    kp_desc = curr_frame.des.copy()
    
    # Select only keypoints that are common with the previous frame
    kpsu = kpsu[idxs_cur]
    kp_desc = kp_desc[idxs_cur]
    
    kdtree = KDTree(kpsu)  # Creating a KDTree from the keypoints
    
    keypoint_dict = {}
    for i, idx in enumerate(idxs_cur):
        key = tuple(kpsu[i])
        keypoint_dict[key] = {
            'desc': tuple(kp_desc[i]),
            'id': idx,  # Original index
            'neighbors': []
        }
    
    if len(kpsu) >= 3:  # Delaunay requires at least 3 points
        simplices = Delaunay(kpsu)
        for simplex in simplices.simplices:
            for i in range(3):
                key_i = tuple(kpsu[simplex[i]])
                for j in range(3):
                    if i != j:
                        key_j = tuple(kpsu[simplex[j]])
                        neighbor_desc = tuple(kp_desc[simplex[j]])
                        if (key_j, neighbor_desc) not in keypoint_dict[key_i]['neighbors']:
                            keypoint_dict[key_i]['neighbors'].append((key_j, neighbor_desc))
    
    return keypoint_dict


def get_common_edges_2():
    return None


def get_common_edges(idxs_ref, idxs_cur, prev_dict, curr_dict, prev_frame, cur_frame):
                    # Keypoint in current frames which are common with the previous frame
                    # print("Common keypoints between the two frames: ", idxs_cur)
                    # print("keys of current dict: ", curr_dict.keys())   
                    # common_cur_kps = [tuple(cur_frame.kps[i]) for i in idxs_cur] # Make cur_frame_kps[i] which is a list of two floatings points to ints before converting to tuple

                    ''' 
                        # Remove duplicates from the keypoints and descriptors
                        kpsu, indices = np.unique(kpsu, axis=0, return_index=True)
                        kp_desc = kp_desc[indices]
                        @Apply same logic for common_cur_kps and common_prev_kps
                    '''
                    common_cur_kps = [tuple(map(int, cur_frame.kpsu[i])) for i in idxs_cur]
                    print("Common keypoints in the current frame: ", len(common_cur_kps))
                    common_prev_kps = [tuple(map(int, prev_frame.kpsu[i])) for i in idxs_ref]
                    print("Common keypoints in the previous frame: ", len(common_prev_kps))

                    # 3D retrieval of the common keypoints 
                    common_cur_kps_3d_pts, common_cur_kps_3d_rgb = cur_frame.unproject_points_3d(idxs_cur, transform_in_world=True)
                    print("Common keypoints in 3D in the current frame: ", len(common_cur_kps_3d_pts))
                    common_prev_kps_3d_pts, common_prev_kps_3d_rgb = prev_frame.unproject_points_3d(idxs_ref, transform_in_world=True)
                    print("Common keypoints in 3D in the previous frame: ", len(common_prev_kps_3d_pts))
                    
                    print("Previous _dict: ", len(prev_dict))
                    print("Current _dict: ", len(curr_dict))

                    common_edges_overall = []
                    # For each common keypoint, find if there are any common edges compared to previous frame
                    for i in range(len(common_cur_kps)):
                        cur_common_idx = i
                        cur_common_kp = common_cur_kps[i]
                        cur_common_kp_3d = common_cur_kps_3d_pts[i]
                        cur_common_kp_rgb = common_cur_kps_3d_rgb[i]
                        cur_common_neighbors = curr_dict[cur_common_kp]['neighbors'] 
                        cur_common_neighbors_idxs = []
                        for neighbor in cur_common_neighbors:
                            # print("Neighbor: ", neighbor)
                            # print("Type of neighbor: ", type(neighbor))
                            neighbor = neighbor[0] # First element of the tuple is the neighbor
                            # Check if the neighbor is in the common keypoints of the current frame - try to find the index of the neighbor in the common keypoints
                            if neighbor in common_cur_kps:
                                neighbor_idx = common_cur_kps.index(neighbor)
                                cur_common_neighbors_idxs.append(neighbor_idx)

                            

                        # print("Common neighbors for the current keypoint which are also in previous frame: ", cur_common_neighbors_idxs)
                        # previous frame common keypoint 
                        prev_common_idx = i 
                        prev_common_kp = common_prev_kps[i]
                        prev_common_kp_3d = common_prev_kps_3d_pts[i]
                        prev_common_kp_rgb = common_prev_kps_3d_rgb[i]
                        prev_common_neighbors = prev_dict[prev_common_kp]['neighbors']
                        prev_common_neighbors_idxs = []
                        for neighbor in prev_common_neighbors:
                            neighbor = neighbor[0]
                            if neighbor in common_prev_kps:
                                neighbor_idx = common_prev_kps.index(neighbor)
                                prev_common_neighbors_idxs.append(neighbor_idx)


                        # Find common edges 
                        common_edges = []
                        for neighbor_idx in cur_common_neighbors_idxs:
                            if neighbor_idx in prev_common_neighbors_idxs:
                                common_edges.append((cur_common_idx, neighbor_idx))

                        # print("Common edges for the current keypoint: ", common_edges)
                        common_edges_overall.extend(common_edges)
                    
                    # print("Common edges overall: ", common_edges_overall)
                    print("Number of common edges overall: ", len(common_edges_overall))

                    # print("Common edges overall: ", common_edges_overall)
                    # Delete duplicate edges 
                    common_edges_overall = list(set(common_edges_overall))
                    print("Number of common edges overall after removing duplicates: ", len(common_edges_overall))
                    # print("Common edges overall after removing duplicates: ", common_edges_overall)

                    # Create a similar list of common edges for the previous frame
                    common_edges_overall_prev = []
                    for i in range(len(common_edges_overall)):
                        edge = common_edges_overall[i]
                        # SAME list to be used for the previous frame
                        common_edges_overall_prev.append(edge) 

                    return common_edges_overall


def convert_frame_to_kdtree_gpu(slam, id=-1):
    ''' 
    GPU-accelerated version using PyTorch for neighbor computation
    Input : SLAM object, id of the frame to be converted to a KDTree
    Output : Dictionary with keypoints as keys and neighbors, desc, id as values
    '''
    
    # Get keypoints and descriptors (original CPU data)
    if id == -1:
        curr_frame = slam.tracking.f_cur
        kpsu = curr_frame.kpsu.copy().astype(int)
        kp_desc = curr_frame.des.copy()
    elif id == -2:
        curr_frame = slam.map.get_frame(-2)
        kpsu = curr_frame.kpsu.copy().astype(int)
        kp_desc = curr_frame.des.copy()

    # Convert to PyTorch tensors and move to GPU
    kpsu_tensor = torch.tensor(kpsu, device='cuda')
    kp_desc_tensor = torch.tensor(kp_desc, device='cuda')

    # CPU-based Delaunay triangulation
    tri = Delaunay(kpsu)
    simplices = tri.simplices

    # Generate all edges from simplices on CPU
    edges = []
    for s in simplices:
        edges.extend([[s[0], s[1]], [s[1], s[2]], [s[0], s[2]]])
    edges = torch.tensor(edges, dtype=torch.long, device='cuda')

    # GPU-accelerated edge processing
    # Remove duplicate edges (undirected)
    sorted_edges, _ = torch.sort(edges, dim=1)
    unique_edges = torch.unique(sorted_edges, dim=0)
    
    # Create bidirectional edges
    bidirectional_edges = torch.cat([unique_edges, unique_edges.flip(1)], dim=0)

    # Group by source nodes using GPU sorting
    sources = bidirectional_edges[:, 0]
    targets = bidirectional_edges[:, 1]
    sorted_sources, indices = torch.sort(sources)
    sorted_targets = targets[indices]

    # Get unique sources and their neighbor counts
    unique_sources, counts = torch.unique_consecutive(sorted_sources, return_counts=True)
    split_targets = torch.split(sorted_targets, counts.tolist())

    # Build neighbor dictionary on CPU
    neighbors_dict = {}
    for src, tgts in zip(unique_sources.cpu().numpy(), split_targets):
        unique_tgts = torch.unique(tgts).cpu().numpy()
        neighbors_dict[src] = [tuple(kpsu[idx]) for idx in unique_tgts]

    # Build result dictionary
    result_dict = {}
    for i, kp in enumerate(kpsu):
        kp_tuple = tuple(kp)
        result_dict[kp_tuple] = {
            'desc': kp_desc[i],
            'id': i,
            'neighbors': [(tuple(kpsu[n]), tuple(kp_desc[n])) 
                         for n in neighbors_dict.get(i, [])]
        }
    print("GPU-accelerated KDTree computation done.")
    return result_dict

def draw_simplicies_on_image(img, dict):
    # Use the kdtree
    for key in dict:
        for neighbor in dict[key]['neighbors']:
            cv2.line(img, key, neighbor[0], (0, 255, 0), 1)
    return img



def filter_delaunay_edges_by_3d_distance(slam, distance_threshold=0.05):
    """
    Filters edges in the Delaunay triangulation based on 3D point distance changes.

    Args:
        slam: The SLAM object.
        distance_threshold: The threshold for the distance change between 3D points.

    Returns:
        A NetworkX graph containing the filtered edges.
    """
    frame = slam.tracking.f_cur  # Get the current frame
    points_2d = frame.kpsu.copy()  # Use undistorted keypoints
    tri = Delaunay(points_2d)  # Compute Delaunay triangulation

    # Project 2D points to 3D in the current frame
    points_3d_cur, valid_3d_mask_cur = frame.unproject_points_3d(
        np.arange(len(frame.kpsu)),
        transform_in_world=True
    )

    # Ensure the boolean mask is correctly applied to filter valid points
    valid_indices = np.where(valid_3d_mask_cur)[0]
    points_3d_cur = points_3d_cur[valid_indices]

    # Get 3D points from the local map
    local_map_points_3d = slam.map.local_map.get_points_as_np()[0]  # Assuming [0] gives the 3D points

    # Create a NetworkX graph
    graph = nx.Graph()
    for simplex in tri.simplices:
        # Get indices of the points forming the edge
        idx1, idx2 = simplex[0], simplex[1]

        # Check if both points have valid 3D correspondences in the current frame and local map
        if (
            idx1 in valid_indices and idx2 in valid_indices and
            frame.points[idx1] is not None and frame.points[idx2] is not None
        ):
            id1 = frame.points[idx1].id
            id2 = frame.points[idx2].id

            # Validate the indices for the local map points
            if id1 < len(local_map_points_3d) and id2 < len(local_map_points_3d):
                # Get corresponding 3D points from the local map
                point_3d_map_1 = local_map_points_3d[id1]
                point_3d_map_2 = local_map_points_3d[id2]

                # Calculate distances
                distance_cur = np.linalg.norm(points_3d_cur[valid_indices.tolist().index(idx1)] - 
                                              points_3d_cur[valid_indices.tolist().index(idx2)])
                distance_map = np.linalg.norm(point_3d_map_1 - point_3d_map_2)

                print(f"Distance change: {abs(distance_cur - distance_map)}")

                # Check if the distance change is within the threshold
                if abs(distance_cur - distance_map) <= distance_threshold:
                    # Add the edge to the graph
                    graph.add_edge(tuple(points_2d[idx1]), tuple(points_2d[idx2]))
            else:
                print(f"Invalid IDs: id1={id1}, id2={id2}")

    return graph  # Return the filtered graph






def filter_delaunay_edges_by_3d_distance_last_frame(slam, distance_threshold=0.001):
    """
    Filters edges in the Delaunay triangulation based on 3D point distance changes
    between the current frame and the last frame.

    Args:
        slam: The SLAM object.
        distance_threshold: The threshold for the distance change between 3D points.

    Returns:
        A NetworkX graph containing the filtered edges.
    """
    frame_cur = slam.tracking.f_cur  # Get the current frame
    frame_last = slam.map.get_frame(-2)  # Get the last frame

    # Check if the last frame is valid
    if frame_last is None:
        print("Error: No last frame available. Returning an empty graph.")
        return nx.Graph()  # Return an empty graph

    points_2d = frame_cur.kpsu.copy()  # Use undistorted keypoints
    tri = Delaunay(points_2d)  # Compute Delaunay triangulation

   # Project 2D points to 3D in the current frame
    points_3d_cur, valid_3d_mask_cur = frame_cur.unproject_points_3d(
        np.arange(len(frame_cur.kpsu)),
        transform_in_world=True
    )
    
    # Apply the mask along the first axis
    points_3d_cur = points_3d_cur[valid_3d_mask_cur, :]  

    # Project 2D points to 3D in the last frame
    points_3d_last, valid_3d_mask_last = frame_last.unproject_points_3d(
        np.arange(len(frame_last.kpsu)),
        transform_in_world=True
    )
    
    # Apply the mask along the first axis
    points_3d_last = points_3d_last[valid_3d_mask_last, :]  

    # Create a NetworkX graph
    graph = nx.Graph()
    for simplex in tri.simplices:
        # Get indices of the points forming the edge
        idx1, idx2 = simplex[0], simplex[1]

        # Check if both points have valid 3D correspondences in the current and last frame
        # AND that the corresponding points in the last frame are the SAME
        if (
            idx1 < len(points_3d_cur) and idx2 < len(points_3d_cur) and
            idx1 < len(points_3d_last) and idx2 < len(points_3d_last) and
            frame_cur.points[idx1] is not None and frame_cur.points[idx2] is not None and
            frame_last.points[idx1] is not None and frame_last.points[idx2] is not None and
            frame_cur.points[idx1] == frame_last.points[idx1] and 
            frame_cur.points[idx2] == frame_last.points[idx2]
        ):
            # Calculate distances
            distance_cur = np.linalg.norm(points_3d_cur[idx1] - points_3d_cur[idx2])
            distance_last = np.linalg.norm(points_3d_last[idx1] - points_3d_last[idx2])

            # Check if the distance change is within the threshold
            if abs(distance_cur - distance_last) <= distance_threshold:
                # Add the edge to the graph
                graph.add_edge(tuple(points_2d[idx1]), tuple(points_2d[idx2]))

    return graph  # Return the filtered graph