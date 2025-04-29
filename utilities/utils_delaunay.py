import numpy as np
from scipy.spatial import Delaunay, KDTree
import cv2
import networkx as nx
import torch
import open3d as o3d

# def get_connected_components(graph):
#   """
#   Divides a graph into its connected components.

#   Args:
#     graph: A networkx graph object.

#   Returns:
#     A list of sets, where each set contains the nodes in a connected component.
#   """
#   return list(nx.connected_components(graph))

# def get_dynamic_components(graph):
#     connected_components = get_connected_components(graph)
#     # Remove the components with the highest number of nodes and return the rest
#     if len(connected_components) > 1:
#         # Find the component with the highest number of nodes
#         max_nodes = max(connected_components, key=len)
#         # Remove the component with the highest number of nodes
#         connected_components.remove(max_nodes)
#     return connected_components


'''
# Example usage:
graph = nx.Graph()
graph.add_edges_from([(1, 2), (2, 3), (4, 5)])  # Example graph with two connected components

connected_components = get_connected_components(graph)
print(connected_components)  # Output: [{1, 2, 3}, {4, 5}]
'''

def delaunay_frame(cur_frame):

    ''' 
    Takes in the current time stamp and slam object and runs delaunay triangulation on the current frame
    
    '''
    # Getting access to the current frame properties after being populated by the SLAM system
    points_2d = cur_frame.kpsu.copy().astype(int)
    tri = Delaunay(points_2d)
    # print("Delaunay Triangulation Done", tri.simplices)
    img = cur_frame.img.copy() 
    for simplex in tri.simplices:
        cv2.line(img, tuple(points_2d[simplex[0]]), tuple(points_2d[simplex[1]]), (0, 255, 0), 1)
        cv2.line(img, tuple(points_2d[simplex[1]]), tuple(points_2d[simplex[2]]), (0, 255, 0), 1)
        cv2.line(img, tuple(points_2d[simplex[2]]), tuple(points_2d[simplex[0]]), (0, 255, 0), 1)
    return img

def delaunay_image_kps(img, kps):
    """ Imput image, keypoints and return the delaunay triangulation of the image with the keypoints """
    # Getting access to the current frame properties after being populated by the SLAM system
    points_2d = kps.copy() # List - convert to numpy array
    points_2d = np.array(points_2d).astype(int)
    tri = Delaunay(points_2d)
    # print("Delaunay Triangulation Done", tri.simplices)
    img = img.copy() 
    for simplex in tri.simplices:
        cv2.line(img, tuple(points_2d[simplex[0]]), tuple(points_2d[simplex[1]]), (0, 255, 0), 1)
        cv2.line(img, tuple(points_2d[simplex[1]]), tuple(points_2d[simplex[2]]), (0, 255, 0), 1)
        cv2.line(img, tuple(points_2d[simplex[2]]), tuple(points_2d[simplex[0]]), (0, 255, 0), 1)
    return img, tri


def delaunay_image_kps_connected_components(img, kps, connected_components):
    """Draw connected components with different colors on the image.
    Assumes the largest component is static background and highlights smaller components
    as potential dynamic objects.
    
    Args:
        img: Input image
        kps: List of keypoint coordinates
        connected_components: List of sets, where each set contains node indices
    """
    img = img.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
             (255, 0, 255), (255, 128, 0), (128, 255, 0), (0, 128, 255)]
    
    # Convert keypoints to integer coordinates
    kps = np.array(kps).astype(int)
    
    # Sort components by size (largest first)
    sorted_components = sorted(connected_components, key=len, reverse=True)
    
    if len(sorted_components) > 1:
        # First component (largest) is assumed to be static background
        static_component = sorted_components[0]
        dynamic_components = sorted_components[1:]
        
        # Draw static component in green
        for node in static_component:
            cv2.circle(img, tuple(kps[node]), 2, (0, 255, 0), -1)
            
        # Draw dynamic components in different colors
        for i, component in enumerate(dynamic_components):
            color = colors[(i+1) % len(colors)]  # Skip green (used for static)
            
            # Draw connections between points in component
            component_list = list(component)
            for j in range(len(component_list)):
                for k in range(j+1, len(component_list)):
                    pt1 = tuple(kps[component_list[j]])
                    pt2 = tuple(kps[component_list[k]])
                    cv2.line(img, pt1, pt2, color, 1)
            
            # Draw points in component
            for node in component:
                cv2.circle(img, tuple(kps[node]), 3, color, -1)
                
            # Add component size label
            if len(component) > 2:  # Only label significant components
                mean_pos = np.mean(kps[list(component)], axis=0).astype(int)
                cv2.putText(img, f'Size: {len(component)}', 
                          tuple(mean_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, color, 1)
    
    return img

def convert_delauany_to_networkx(tri):
    ''' 
    Takes in the delaunay triangulation and converts it to a networkx graph object
    
    '''
    graph = nx.Graph()
    for simplex in tri.simplices:
        graph.add_edges_from([(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])])

    return graph

def delaunay_triangulation(f_cur):
    tri = Delaunay(f_cur.kpsu.copy().astype(int))
    graph = convert_delauany_to_networkx(tri)
    return graph


