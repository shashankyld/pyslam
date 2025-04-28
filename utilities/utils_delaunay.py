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

def delaunay_image(cur_frame):

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


