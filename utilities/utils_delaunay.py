import numpy as np
from scipy.spatial import Delaunay
import cv2
import networkx as nx

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