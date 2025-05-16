# Given a point cloud, find a plane that best fits the points using RANSAC, remove the outliers, then estimate the stddev of the inliers.
import numpy as np
import open3d as o3d
from typing import Tuple, List

class PlaneFitting:
    def __init__(self, point_cloud):
        self.point_cloud = point_cloud
        self.inliers = None
        self.outliers = None
        self.plane_model = None
        self.stddev = None

    def fit_plane(self, distance_threshold=0.05, ransac_n=3, num_iterations=1000) -> Tuple[np.ndarray, List]:
        """
        Fit a plane to the point cloud using RANSAC
        
        Args:
            distance_threshold: Maximum distance a point can be from the plane to be considered an inlier
            ransac_n: Number of points to sample for each RANSAC iteration
            num_iterations: Number of RANSAC iterations
            
        Returns:
            Tuple containing plane equation coefficients [a, b, c, d] and list of inlier indices
        """
        # Convert point cloud to Open3D format if it's a numpy array
        if isinstance(self.point_cloud, np.ndarray):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
        else:
            pcd = self.point_cloud
            
        # Apply RANSAC to fit a plane
        plane_model, inlier_indices = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        # Store results
        self.plane_model = plane_model  # [a, b, c, d] in ax + by + cz + d = 0
        self.inliers = np.asarray(pcd.points)[inlier_indices]
        
        # Find outliers (points not in inliers)
        all_indices = set(range(len(np.asarray(pcd.points))))
        outlier_indices = list(all_indices - set(inlier_indices))
        self.outliers = np.asarray(pcd.points)[outlier_indices]
        
        return plane_model, inlier_indices

    def remove_outliers(self, distance_threshold=0.01):
        """
        Remove outliers from the point cloud based on the fitted plane model
        
        Args:
            distance_threshold: Maximum distance a point can be from the plane to be considered an inlier
            
        Returns:
            Numpy array of inlier points
        """
        if self.plane_model is None:
            self.fit_plane(distance_threshold=distance_threshold)
            
        return self.inliers

    def estimate_stddev(self):
        """
        Estimate the standard deviation of the inliers' distances to the fitted plane
        
        Returns:
            Standard deviation of inlier points' distances to the plane
        """
        if self.inliers is None or self.plane_model is None:
            self.fit_plane()
            
        # Get the plane equation parameters
        a, b, c, d = self.plane_model
        norm = np.sqrt(a**2 + b**2 + c**2)
        
        # Calculate the distance from each inlier point to the plane
        distances = []
        for point in self.inliers:
            # Distance formula for a point to a plane: |ax + by + cz + d| / sqrt(a² + b² + c²)
            distance = abs(np.dot(point, [a, b, c]) + d) / norm
            distances.append(distance)
        
        # Calculate standard deviation of distances
        self.stddev = np.std(distances)
        print(f"Standard deviation of inliers' distances to the plane: {self.stddev}")
        return self.stddev

    def visualize(self, plane_size=1.0, plane_color=[0.8, 0.8, 0.8], alpha=0.5):
        """
        Visualize the point cloud with inliers in green, outliers in red, and the fitted plane as transparent
        
        Args:
            plane_size: Size of the plane to visualize
            plane_color: RGB color of the plane [r, g, b] where r,g,b are in [0,1]
            alpha: Transparency of the plane (between 0 and 1)
        """
        if self.inliers is None or self.outliers is None or self.plane_model is None:
            self.fit_plane()
            
        # Create point clouds for inliers and outliers
        inlier_cloud = o3d.geometry.PointCloud()
        inlier_cloud.points = o3d.utility.Vector3dVector(self.inliers)
        inlier_cloud.paint_uniform_color([0, 1, 0])  # Green
        
        outlier_cloud = o3d.geometry.PointCloud()
        outlier_cloud.points = o3d.utility.Vector3dVector(self.outliers)
        outlier_cloud.paint_uniform_color([1, 0, 0])  # Red
        
        # Create a mesh for the plane
        a, b, c, d = self.plane_model
        
        # Get the center of the point cloud to place the plane
        points = np.vstack((self.inliers, self.outliers))
        center = np.mean(points, axis=0)
        
        # Create a coordinate system at the center point
        # with z-axis along plane normal
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        
        # Find two orthogonal vectors in the plane
        if np.abs(normal[0]) > np.abs(normal[1]):
            v1 = np.array([-normal[2], 0, normal[0]]) / np.sqrt(normal[0]**2 + normal[2]**2)
        else:
            v1 = np.array([0, -normal[2], normal[1]]) / np.sqrt(normal[1]**2 + normal[2]**2)
        
        v2 = np.cross(normal, v1)
        v1 = v1 * plane_size
        v2 = v2 * plane_size
        
        # Create vertices for plane mesh
        vertices = [
            center - v1 - v2,
            center + v1 - v2,
            center + v1 + v2,
            center - v1 + v2
        ]
        
        # Create triangular mesh for the plane
        plane_mesh = o3d.geometry.TriangleMesh()
        plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        plane_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
        plane_mesh.compute_vertex_normals()
        plane_mesh.paint_uniform_color(plane_color)
        
        # Create a material for transparency
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        material.base_color = plane_color + [alpha]  # RGBA
        
        # Visualize
        o3d.visualization.draw_geometries_with_custom_animation([inlier_cloud, outlier_cloud, plane_mesh],
                                                               window_name="Plane Fitting Visualization")