import numpy as np 
import json
from scipy.optimize import minimize
from math import asin, atan, pi, tan, sin, cos
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def distance(point1, point2):
    R = 6371000  # Earth radius in kilometers
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = np.radians(point1)
    lat2, lon2 = np.radians(point2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c

    return distance

def displacement_vector(Point, cam):
    # Calculate the displacement vector in meters
    delta_x = distance(Point, np.array([Point[0], cam[1]])) * (1 if Point[1] > cam[1] else -1)
    delta_y = distance(Point, np.array([cam[0], Point[1]])) * (1 if Point[0] > cam[0] else -1)
    return np.array([delta_x, delta_y])


def angle_between(vector1, vector2):
    dot_prod = np.dot(vector1, vector2)
    mag1 = np.linalg.norm(vector1)
    mag2 = np.linalg.norm(vector2)

    # Calculate the cosine of the angle
    cos_theta = dot_prod / (mag1 * mag2)

    # Calculate the angle in radians
    radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return radians

def Poly_fit(degree, Angles, Radiuses):

    # Define the polynomial function
    def polynomial(x, *odd_coefficients):
        zeros_array = np.zeros(len(odd_coefficients))
        coefficients = np.insert(odd_coefficients, np.arange(1, len(odd_coefficients)+1), zeros_array)
        return np.polyval(coefficients, x)

    # Define the objective function to minimize (least squares)
    def objective(coefficients, x, y):
        return np.sum((polynomial(x, *coefficients) - y)**2)

    # Initial guess for polynomial coefficients
    initial_guess = np.ones((degree + 1)//2) 

    # Perform the optimization
    result = minimize(objective, initial_guess, args=(Angles, Radiuses), method='L-BFGS-B')

    # Extract the optimized coefficients
    optimized_coefficients = result.x
    
    # Evaluate the polynomial at the optimized coefficients
    sum, Max_error, index = (0, 0, 0)
    for i in range(len(Angles)):
        diff = polynomial(Angles[i], *optimized_coefficients) - Radiuses[i]
        error = diff / Radiuses [i] * 100 if not(Radiuses[i] == 0) else 0
        Max_error, index = (error, i) if error > Max_error else (Max_error, index)
        sum += abs(error)
        
    print(f"Average camera calibration radius error: {sum/len(Angles):.1f} %")
    print(f"Maximum camera calibration radius error: {Max_error:.1f} % for point {index}")

    return optimized_coefficients

def Plot3D(points):
    # Separate the points into x, y, and z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Create grid data for x and y
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]

    # Interpolate the z values on the grid
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')

    # Add color bar which maps values to colors
    fig.colorbar(surf)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

class TriangularizationCurveFit:
    def __init__(self, points):
        """
        Initialize the TriangularizationCurveFit object with the given points.

        Args:
        - points: A list or array of 3D points in the format [(x1, y1, z1), (x2, y2, z2), ...].
        """
        self.points = np.array(points)
        self.tri = Delaunay(self.points[:, :2])  # Triangulate points based on (x, y) coordinates

    def fit(self):
        """
        Perform the triangularization curve fitting.
        
        Returns:
        - coeffs: Coefficients of the linear fits for each triangle.
        """
        coeffs = []
        for simplex in self.tri.simplices:
            x = self.points[simplex, 0]
            y = self.points[simplex, 1]
            z = self.points[simplex, 2]

            # Fit a plane (z = a*x + b*y + c) to the triangle
            A = np.vstack([x, y, np.ones_like(x)]).T
            a, b, c = np.linalg.lstsq(A, z, rcond=None)[0]
            coeffs.append((a, b, c))

        return coeffs

    def compute_z(self, coord):
        """
        Compute the z-value for the given (x, y) pair using the fitted piecewise linear curve.

        Args:
        - x: x-coordinate.
        - y: y-coordinate.

        Returns:
        - z: Interpolated z-value.
        """
        simplex_index = self.tri.find_simplex(coord)
        if simplex_index == -1:
            return None
        else:
            a, b, c = self.fit()[simplex_index]
            z = a * coord[0] + b * coord[1] + c
            return z

class Transform_Coordinates:
    def __init__(self, H, Polynomial_degree, Landmark_file):
        self.earth_radius = 6371e3
        self.H = H
        self._Calibrate(Polynomial_degree, Landmark_file)

    def _Calibrate(self, degree, Landmark_file):

        # Read JSON data from file
        with open(Landmark_file, 'r') as file:
            json_data = json.load(file)
                                                
        # Access the dictionary
        gps_points     = json_data['GPSPoints']
        fisheye_points = json_data['FisheyePoints']
        birdeye_points = json_data['BirdeyePoints']

        self.Lat = np.radians(gps_points['0'][0]) # Latitude of the camera in radians
        self.Lon = np.radians(gps_points['0'][1]) # Longitude of the camera in radians
        self.Birdeye_cam = np.array(birdeye_points['0']) # Birdeye camera location in pixels
        self.Fisheye_camera_pixels = np.array(fisheye_points[str(0)]) # Fisheye camera location in pixels 
        ref_point_num = len(gps_points)

        Distances_from_camera = [round(distance(gps_points[str(i)],gps_points[str(0)]), 1) 
                                for i in range(ref_point_num)] # Distance from camera to each landmark in meters in real world
        
        Angles = np.array([atan(distace/self.H) for distace in Distances_from_camera]) # Incoming ray angle from each landmark to the camera

        Radiuses = np.array([np.linalg.norm(np.array(fisheye_points[str(i)]) - self.Fisheye_camera_pixels)
                            for i in range(ref_point_num)]) # Distance from camera to each landmark in pixels inside fisheye image
        
        optimized_coefficients = Poly_fit(degree, Angles, Radiuses)
        print(f"Polynomial coefficients: {optimized_coefficients}")
        # Modify the polynomial coefficients to include the even powers
        zeros_array = np.zeros(len(optimized_coefficients))
        self.poly = np.insert(optimized_coefficients, np.arange(1, len(optimized_coefficients)+1), zeros_array)
        self.compute_altitude(Radiuses, Distances_from_camera)

        self.Fisheye_vectors = np.array([self.defisheye(fisheye_points[str(i)], 'None-iterative') for i in range(1, ref_point_num)])
        # concatinate self.Fisheye_vectors with self.heights[1:] to get the 3D fisheye vectors
        Elevation_Data = np.concatenate((self.Fisheye_vectors, np.expand_dims(self.heights[1:], axis=1)), axis=1)
        self.tcf = TriangularizationCurveFit(Elevation_Data)
        self.Fisheye_vectors_Corrected = [self.defisheye(fisheye_points[str(i)], 'Iterative') for i in range(1, ref_point_num)]
        GPS_vectors     = [displacement_vector(gps_points[str(i)], gps_points[str(0)]) for i in range(1, ref_point_num)]
        Vector_Angles   = [angle_between(self.Fisheye_vectors[i], GPS_vectors[i]) for i in range(ref_point_num -1)]
        filtered_Angles = [angle for i, angle in enumerate(Vector_Angles) if Distances_from_camera[i] > 20]
        Rotation_Angle  = -np.mean(filtered_Angles)
        self.rotation_matrix = np.array([[cos(Rotation_Angle), -sin(Rotation_Angle)],
                                        [sin(Rotation_Angle), cos(Rotation_Angle)]])
        Birdeye_vectors = [np.array(birdeye_points[str(i)]) - np.array(birdeye_points[str("0")]) for i in range(1, ref_point_num)]
        expansions      = [np.linalg.norm(Birdeye_vectors[i]) / np.linalg.norm(self.Fisheye_vectors_Corrected[i]) for i in range(ref_point_num - 1)]
        self.expansion  = np.mean(expansions)   

    def compute_altitude(self, Radiuses, Distances_from_camera):
        self.heights = np.ones(len(Radiuses)) * self.H
        for i, radius in enumerate(Radiuses):
            Poly = self.poly.copy() 
            Poly[-1] -= radius
            roots = np.roots(Poly)
            real_roots = roots[np.isreal(roots)].real
            if real_roots.any():
                final_root = [root for root in real_roots if 0 < root < pi/2]
                if final_root:
                    self.heights[i] = Distances_from_camera[i] / tan(min(final_root))



    def defisheye(self, Fisheye_pixel_point, Method, h=0):
        R = np.array(Fisheye_pixel_point) - self.Fisheye_camera_pixels
        r = np.linalg.norm(R)
        R = R/r 
        R[1] *= -1
        Poly = self.poly.copy() 
        Poly[-1] -= r
        roots = np.roots(Poly)
        real_roots = roots[np.isreal(roots)].real
        if real_roots.any():
            final_root = [root for root in real_roots if 0 < root < pi/2]
            if final_root:
                Correction = (self.H - h/2)/self.H
                theta = min(final_root)
                rectified_coord = Correction * self.H * tan(theta) * R 
                if Method == 'Iterative':
                        self.Elevation = self.tcf.compute_z(rectified_coord)
                        if self.Elevation is None:
                            return np.array([])
                        else:
                            Correction = (self.Elevation - h/2)/self.Elevation
                            rectified_coord =  Correction * self.Elevation * tan(theta) * R 
            
                return rectified_coord 
            else:
                return np.array([])
        else:
            return np.array([])
    
    def fisheye(self, coord):
        if len(np.shape(coord)) == 1:
            coord = np.expand_dims(coord, axis = 0)   
        FishEyedPos = np.empty((0,2))
        for point in coord:
            d = np.linalg.norm(point)
            unit_vector = point/d 
            r = np.polyval(self.poly, atan(d/self.H)) * unit_vector
            r[1] *= -1
            r += self.Fisheye_camera_pixels
            FishEyedPos = np.vstack([FishEyedPos, r.astype(int)])   
  
        return FishEyedPos
    
    def GPS(self, displacement_vector):
        # Extract displacement vector components
        delta_x, delta_y = displacement_vector

        # Calculate new latitude and longitude
        new_lat = self.Lat + (delta_y / self.earth_radius)
        new_lon = self.Lon + (delta_x / (self.earth_radius * cos(self.Lat)))
     
        # Ensure longitude is within the valid range (-180 to 180 degrees)
        new_lon = (new_lon + 180) % 360 - 180

        # Convert latitude and longitude back to degrees
        new_lat, new_lon = np.degrees(new_lat), np.degrees(new_lon)

        return np.array([new_lat, new_lon])
    
    def BirdEye(self, Rotated_coord):
        coord = Rotated_coord.copy()
        coord[1] *= -1
        Birdeye_point = self.Birdeye_cam + self.expansion * coord
        return Birdeye_point.astype(int)

    
    def Transform(self, Rectified_coord):
        if Rectified_coord.any():
            Rotated_coord   = np.dot(self.rotation_matrix, Rectified_coord)
            GPS_coord       = self.GPS(Rotated_coord)
            Birdeye_coord   = self.BirdEye(Rotated_coord) 
            return GPS_coord, Birdeye_coord, Rotated_coord
        else:
            empty = np.array([])
            return empty, empty, empty
        

        

Transformer = Transform_Coordinates(H=10, Polynomial_degree=5, Landmark_file='Positioning/cfg/Calibration.json') 
