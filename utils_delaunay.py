from scipy.spatial import Delaunay
import numpy as np

def adjaceny_matrix(tri):
    """
    Compute simplicies, and then create a adjacency matrix with axis as indices
    fill this matrix with 1 for each pair of points that share an edge - get this from the simplicies
    """
    simplices = tri.simplices
    adj = np.zeros((tri.npoints, tri.npoints))
    for s in simplices:
        adj[s[0], s[1]] = 1
        adj[s[1], s[0]] = 1
        adj[s[0], s[2]] = 1
        adj[s[2], s[0]] = 1
        adj[s[1], s[2]] = 1
        adj[s[2], s[1]] = 1
    return adj

def get_common_edges(tri1, tri2):
    """
    Get the common edges between two delaunay triangulations
    """
    adj1 = adjaceny_matrix(tri1)
    adj2 = adjaceny_matrix(tri2)
    return adj1 * adj2

def get_dynamic_edges(points1 : np.ndarray, points2 : np.ndarray):
    """
    Get the dynamic edges between two delaunay triangulations,
    for each common edge - check lengths of the edges in both triangulations
    if the lengths are similar, then the edge is considered static else dynamic
    """

    # Change data type to int 
    points1 = points1.astype(int)
    points2 = points2.astype(int)

    tri1 = Delaunay(points1)
    tri2 = Delaunay(points2)

    common_edges = get_common_edges(tri1, tri2)
    edges = []
    # Since there will be duplicates, we just search for the upper triangular part of the matrix
    for i in range(common_edges.shape[0]):
        for j in range(i+1, common_edges.shape[1]):
            if common_edges[i, j] == 1:
                # Check if the edge is dynamic
                if np.linalg.norm(points1[i] - points1[j]) - np.linalg.norm(points2[i] - points2[j]) > 0.15* np.linalg.norm(points1[i] - points1[j]):
                    edges.append((i, j))
    return edges

def get_dynamic_edges_2(points1 : np.ndarray, points2 : np.ndarray):
    """
    Get the dynamic edges between two delaunay triangulations,
    for each common edge - check lengths of the edges in both triangulations
    if the lengths are similar, then the edge is considered static else dynamic
    There can be many solutions in 3D that will have same edge length in 2D, for example 
    a point travelling on a sphere with centre at a neighbouring static point comes out static in the previous 
    implentation. So, also checks for the angle between the edges in 2D, only as a second condition as camera motion also 
    changes angles between static planes.
    """
    
    # Change data type to int 
    points1 = points1.astype(int)
    points2 = points2.astype(int)
    
    tri1 = Delaunay(points1)
    tri2 = Delaunay(points2)
    
    common_edges = get_common_edges(tri1, tri2)
    edges = []
    # Since there will be duplicates, we just search for the upper triangular part of the matrix
    for i in range(common_edges.shape[0]):
        for j in range(i+1, common_edges.shape[1]):
            if common_edges[i, j] == 1:
                # Check if the edge is dynamic
                if np.linalg.norm(points1[i] - points1[j]) - np.linalg.norm(points2[i] - points2[j]) > 0.08* np.linalg.norm(points1[i] - points1[j]):
                    # Check if the angle between the edges is different
                    edge1 = points1[j] - points1[i]
                    edge2 = points2[j] - points2[i]
                    if np.dot(edge1, edge2) / (np.linalg.norm(edge1) * np.linalg.norm(edge2)) < 0.99:
                        edges.append((i, j))
    return edges


## LENGTH PARAM - Depends on the slowest moving object that you want to detect
## Angle param - Highly dependent on camera attitude motion, Ideas like increasing trend, decreasing trends also dont work, 
# maybe if I plot angle changes, I might find dynamic objects out of distribution
## Maybe params are best suited only for a specific type of motion, and on specific dataset
## How to make it more general?