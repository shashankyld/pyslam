
import numpy as np
import cv2
from scipy.spatial import Delaunay


def remove_duplicates_from_index_arrays(idxs_ref, idxs_cur):
    """
    Removes duplicates from idxs_ref and corresponding elements from idxs_cur,
    then removes any remaining duplicates from idxs_cur and corresponding elements from idxs_ref.
    Handles empty idxs_cur.

    Args:
        idxs_ref: Indices of reference frame keypoints.
        idxs_cur: Indices of current frame keypoints.

    Returns:
        Updated idxs_ref, idxs_cur.
    """

    if len(idxs_cur) == 0:  # Handle empty idxs_cur
        print("Warning: idxs_cur is empty. Returning original arrays.")
        return idxs_ref, idxs_cur

    # 1. Identify and Remove Duplicates in idxs_ref
    unique_idxs_ref, unique_indices = np.unique(idxs_ref, return_index=True)
    duplicate_indices = np.setdiff1d(np.arange(len(idxs_ref)), unique_indices)

    mask = np.ones(len(idxs_ref), dtype=bool)
    mask[duplicate_indices] = False

    idxs_ref_updated = idxs_ref[mask]
    idxs_cur_updated = idxs_cur[mask]

    # 2. Identify and Remove Remaining Duplicates in idxs_cur
    unique_idxs_cur, unique_indices_cur = np.unique(idxs_cur_updated, return_index=True)
    duplicate_indices_cur = np.setdiff1d(np.arange(len(idxs_cur_updated)), unique_indices_cur)

    mask_cur = np.ones(len(idxs_cur_updated), dtype=bool)
    mask_cur[duplicate_indices_cur] = False

    idxs_cur_updated = idxs_cur_updated[mask_cur]
    idxs_ref_updated = idxs_ref_updated[mask_cur]

    return idxs_ref_updated, idxs_cur_updated


def delaunay_with_kps(frame, idxs):
    kps = frame.kps
    if len(idxs) == 0:
        print("Warning: No keypoints available for Delaunay triangulation.")
        return None

    kps = kps[idxs]  # Filter keypoints

    if len(kps) < 3:  # Delaunay needs at least 3 points
        print("Warning: Not enough points for triangulation.")
        return None
    
    # Convert to int if necessary
    kps = kps.astype(np.int32)

    # Perform Delaunay triangulation
    tri = Delaunay(kps) # <scipy.spatial._qhull.Delaunay object at 0x7eff1746d040>
    tri_indices = tri.simplices # [[215 590 414], [213 212 414]...] # theses are the indices of the points in kps that form the triangles
    tri_vertices = kps[tri_indices] 
    '''
    # These are the vertices of the triangles in the image in pixel coordinates
    tri_vertices =
    [
        [
            [ 57 443]
            [ 70 450]
            [ 70 451]
        ]

        [
            [ 71 451]
            [ 86 456]
            [ 70 451]
        ]
    ]
    '''

    # Draw triangles
    img = frame.img.copy()
    for tri_vert in tri_vertices:
        tri_vert = np.array(tri_vert, dtype=np.int32)  # Ensure integer format
        cv2.polylines(img, [tri_vert], isClosed=True, color=(0, 255, 0), thickness=1)

    print("Delaunay triangulation completed.")
    print("tri", tri)
    print("tri_indices", tri_indices)
    print("tri_vertices", tri_vertices)
    return tri_indices, tri_vertices, img


    

def convert_frame_to_delaunay_dict(frame, idxs):
    kpsu = frame.kpsu[idxs] # [[ 97.02989  306.17648 ], [ 45.978912 326.03632 ], ...]
    kps  = frame.kps[idxs]  # [[ 57. 321.],  [ 71. 299.], ...]
    kpsn = frame.kpsn[idxs] # [[-0.46763112  0.15167243],  [-0.44217141  0.1117238 ], ...]
    kps_depth = frame.depths[idxs] # [1.947 1.926 1.756 ...]
    delaunay_with_kps_ = delaunay_with_kps(frame, idxs)
    return {
        'frame': frame,
        'idxs': idxs,
        'kpsu': kpsu,
        'kps': kps,
        'kpsn': kpsn,
        'kps_depth': kps_depth,
        'delaunay_with_kps': delaunay_with_kps_
    }
    

def delaunay_visualization(prev_delaunay_img, curr_delaunay_img):
    # Horizontally stack the images
    img = np.hstack((prev_delaunay_img, curr_delaunay_img))
    cv2.imshow("Delaunay Triangulation", img)
    cv2.waitKey(2)  


# def get_common_edges(prev_frame_dict, cur_frame_dict):
#     # Check if Delaunay triangulation data exists for both frames
#     prev_delaunay_data = prev_frame_dict.get('delaunay_with_kps')
#     curr_delaunay_data = cur_frame_dict.get('delaunay_with_kps')

#     prev_idxs = prev_frame_dict.get('idxs')
#     curr_idxs = cur_frame_dict.get('idxs')

#     print("Prev idxs:", len(prev_idxs))
#     print("Curr idxs:", len(curr_idxs))

#     # Create a dict with prev idxs as keys and curr idxs as values
#     idxs_dict = dict(zip(prev_idxs, curr_idxs))
#     # print("Idxs dict:", idxs_dict.keys())
#     print("len Idxs dict:", len(idxs_dict))



#     if prev_delaunay_data is None or curr_delaunay_data is None:
#         return []
    
#     # Extract triangle indices from both frames
#     prev_tri_indices = prev_delaunay_data[0]
#     curr_tri_indices = curr_delaunay_data[0]

#     # print("Prev tri indices:", prev_tri_indices)
#     # print("Curr tri indices:", curr_tri_indices)
    
#     # Helper function to extract edges from triangle indices
#     def extract_edges(tri_indices):
#         edges = set()
#         for simplex in tri_indices:
#             # Generate all three edges of the triangle and store as sorted tuples
#             a, b, c = simplex
#             edges.add(tuple(sorted((a, b))))
#             edges.add(tuple(sorted((b, c))))
#             edges.add(tuple(sorted((c, a))))
#         return edges
    
    
#     # Get edges for previous and current frames
#     prev_edges = extract_edges(prev_tri_indices)
#     curr_edges = extract_edges(curr_tri_indices)

#     print("Length of prev edges:", len(prev_edges))
#     print("Length of curr edges:", len(curr_edges))
    
#     # print("Prev edges:", prev_edges)
#     # print("######################################")
#     # print("######################################")
#     # print("######################################")
#     # print("######################################")
#     # print("Curr edges:", curr_edges)
    
#     # For each edge in previous frame, create a probable edge in the current frame using idxs_dict
#     common_edges = set()
#     vertices_prev = set()
#     for edge in prev_edges: 
#         a, b = edge
#         vertices_prev.add(a)
#         vertices_prev.add(b)

#     print("Vertices prev:", len(vertices_prev))
#     # for every vertex in the vertices_prev, check if it is in the idxs_dict 
#     count = 0
#     for vertex in vertices_prev:
#         if vertex in idxs_dict:
#             True
#             count += 1
#     print("Count:", count)

#     # # Now intersect the two sets of fake and real edges to get the common edges
#     # common_edges = common_edges.intersection(curr_edges)
#     print("Length of common edges:", len(common_edges))


            

    # vertices_prev = set()
    # for edge in prev_edges:
    #     a, b = edge
    #     if a in idxs_dict and b in idxs_dict:
    #         vertices_prev.add(a)
    #         vertices_prev.add(b)

    #     fake_edge_a = idxs_dict.get(a)
    #     fake_edge_b = idxs_dict.get(b)
        
    # return list(common_edges)


def get_common_edges(prev_frame_dict, cur_frame_dict):
    prev_delaunay_data = prev_frame_dict.get('delaunay_with_kps')
    curr_delaunay_data = cur_frame_dict.get('delaunay_with_kps')

    prev_idxs = prev_frame_dict.get('idxs')
    curr_idxs = cur_frame_dict.get('idxs')

    if prev_delaunay_data is None or curr_delaunay_data is None:
        return []

    idxs_dict = dict(zip(prev_idxs, curr_idxs))
    reverse_curr_dict = {orig_idx: pos for pos, orig_idx in enumerate(curr_idxs)}

    prev_tri_indices = prev_delaunay_data[0]
    curr_tri_indices = curr_delaunay_data[0]

    def extract_edges(tri_indices):
        edges = set()
        for simplex in tri_indices:
            a, b, c = simplex
            edges.add(tuple(sorted((a, b))))
            edges.add(tuple(sorted((b, c))))
            edges.add(tuple(sorted((c, a))))
        return edges

    prev_edges = extract_edges(prev_tri_indices)
    curr_edges = extract_edges(curr_tri_indices)

    common_edges = set()

    for edge in prev_edges:
        a_prev, b_prev = edge
        orig_a_prev = prev_idxs[a_prev]
        orig_b_prev = prev_idxs[b_prev]

        orig_a_curr = idxs_dict.get(orig_a_prev)
        orig_b_curr = idxs_dict.get(orig_b_prev)

        if orig_a_curr is None or orig_b_curr is None:
            continue

        pos_a_curr = reverse_curr_dict.get(orig_a_curr)
        pos_b_curr = reverse_curr_dict.get(orig_b_curr)

        if pos_a_curr is None or pos_b_curr is None:
            continue

        sorted_edge = tuple(sorted((pos_a_curr, pos_b_curr)))
        if sorted_edge in curr_edges:
            common_edges.add((orig_a_prev, orig_b_prev, orig_a_curr, orig_b_curr))
    
    print("Length of prev idxs:", len(prev_idxs))
    print("Length of curr idxs:", len(curr_idxs))
    print("Length of prev edges:", len(prev_edges))
    print("Length of common edges:", len(common_edges))
    print("Common edges:", common_edges)
    return list(common_edges)

def draw_common_edges(prev_frame, curr_frame,common_edges):
    # Horizontally stack the images
    img = np.hstack((prev_frame.img, curr_frame.img))
    for edge in common_edges:
        a_prev, b_prev, a_curr, b_curr = edge
        a_prev = prev_frame.kps[a_prev].astype(np.int32)
        b_prev = prev_frame.kps[b_prev].astype(np.int32)
        a_curr = curr_frame.kps[a_curr].astype(np.int32)
        b_curr = curr_frame.kps[b_curr].astype(np.int32)
        cv2.line(img, tuple(a_prev), tuple(b_prev), (0, 255, 0), 1) 
        cv2.line(img, tuple(a_curr + [prev_frame.img.shape[1], 0]), tuple(b_curr + [prev_frame.img.shape[1], 0]), (0, 255, 0), 1)
    cv2.imshow("Common Edges", img)
    cv2.waitKey(2)


def get_dynamic_edges(curr_frame, prev_frame, common_edges, threshold=0.1):
    """ 
    Returns the dynamic edges between the current and previous frame.
    All the edge lengths are calculated in the current frame and the previous frame. in the world frame/sensor frame- we can directly check in sensor frame as we have rgbd data
    for all the edges in the common_edges, the ones that change their length are considered as dynamic edges. by a margin of 0.1
    """
    dynamic_edges = []
    for edge in common_edges:
        a_prev, b_prev, a_curr, b_curr = edge
        a_prev_depth = prev_frame.depths[a_prev]
        b_prev_depth = prev_frame.depths[b_prev]
        a_curr_depth = curr_frame.depths[a_curr]
        b_curr_depth = curr_frame.depths[b_curr]

        prev_edge_length = np.linalg.norm(a_prev_depth - b_prev_depth)
        curr_edge_length = np.linalg.norm(a_curr_depth - b_curr_depth)

        if abs(curr_edge_length - prev_edge_length) > threshold:
            dynamic_edges.append(edge)
    return dynamic_edges

def draw_dynamic_edges(prev_frame, curr_frame, dynamic_edges):
    # Horizontally stack the images
    img = np.hstack((prev_frame.img, curr_frame.img))
    for edge in dynamic_edges:
        a_prev, b_prev, a_curr, b_curr = edge
        a_prev = prev_frame.kps[a_prev].astype(np.int32)
        b_prev = prev_frame.kps[b_prev].astype(np.int32)
        a_curr = curr_frame.kps[a_curr].astype(np.int32)
        b_curr = curr_frame.kps[b_curr].astype(np.int32)
        cv2.line(img, tuple(a_prev), tuple(b_prev), (0, 0, 255), 1)
        cv2.line(img, tuple(a_curr + [prev_frame.img.shape[1], 0]), tuple(b_curr + [prev_frame.img.shape[1], 0]), (0, 0, 255), 1)
    cv2.imshow("Dynamic Edges", img)
    cv2.waitKey(2)

import networkx as nx


def get_static_edges(common_edges, dynamic_edges):
    """
    Returns the static edges between the current and previous frame.
    """
    common_edges_set = set(common_edges)
    dynamic_edges_set = set(dynamic_edges)
    static_edges = common_edges_set - dynamic_edges_set
    return list(static_edges)

def draw_static_edges(prev_frame, curr_frame, static_edges):
    # Horizontally stack the images
    img = np.hstack((prev_frame.img, curr_frame.img))
    for edge in static_edges:
        a_prev, b_prev, a_curr, b_curr = edge
        a_prev = prev_frame.kps[a_prev].astype(np.int32)
        b_prev = prev_frame.kps[b_prev].astype(np.int32)
        a_curr = curr_frame.kps[a_curr].astype(np.int32)
        b_curr = curr_frame.kps[b_curr].astype(np.int32)
        cv2.line(img, tuple(a_prev), tuple(b_prev), (255, 0, 0), 1)
        cv2.line(img, tuple(a_curr + [prev_frame.img.shape[1], 0]), tuple(b_curr + [prev_frame.img.shape[1], 0]), (255, 0, 0), 1)
    cv2.imshow("Static Edges", img)
    cv2.waitKey(2)



## UTILS FOR CONVERTING EDGES TO GRAPH and finding connected components after removing dynamic edges from the graph



import networkx as nx

def get_connected_components_from_edges(common_edges, prev_frame, curr_frame):
    # Create an empty graph
    G = nx.Graph()
    
    # Add nodes to the graph from keypoints (you could use indices of keypoints as nodes)
    for idx in range(len(prev_frame.kps)):
        G.add_node(idx)  # Adding nodes for previous frame keypoints
    for idx in range(len(curr_frame.kps)):
        G.add_node(len(prev_frame.kps) + idx)  # Adding nodes for current frame keypoints
    
    # Add edges between corresponding keypoints
    for edge in common_edges:
        a_prev, b_prev, a_curr, b_curr = edge
        # Add edges between matching keypoints: one for previous and one for current
        G.add_edge(a_prev, a_curr + len(prev_frame.kps))  # From prev to curr
        G.add_edge(b_prev, b_curr + len(prev_frame.kps))  # From prev to curr
    
    # Find the connected components
    connected_components = list(nx.connected_components(G))
    
    return connected_components

# Example usage
common_edges = [(0, 1, 2, 3), (4, 5, 6, 7)]  # Example common edges
# Assuming prev_frame and curr_frame are available
# connected_components = get_connected_components_from_edges(common_edges, prev_frame, curr_frame)

# Print the connected components
# print("Connected Components:", connected_components)

def draw_connected_components(prev_frame, curr_frame, connected_components):
    # Horizontally stack the images
    img = np.hstack((prev_frame.img, curr_frame.img))
    
    # Draw the connected components
    for component in connected_components:
        for idx in component:
            if idx < len(prev_frame.kps):
                cv2.circle(img, tuple(prev_frame.kps[idx].astype(np.int32)), 3, (0, 255, 0), -1)
            else:
                cv2.circle(img, tuple(curr_frame.kps[idx - len(prev_frame.kps)].astype(np.int32) + [prev_frame.img.shape[1], 0]), 3, (0, 255, 0), -1)
    
    cv2.imshow("Connected Components", img)
    cv2.waitKey(2)


