
import cv2
from math import tan, atan, exp, pi, sin, asin, cos
from Positioning.Calibration import Transformer
from filterpy.kalman import KalmanFilter
import numpy as np

class Tracks:
    detla_t = 0.1
    all = []
    Names = {0:'Person', 1:'Bike', 2:'Car', 3:'Motorcycle', 4:'Bus', 5:'Truck'}
    ObjectDim = {
        0:[[0.5, 0.6, 1.8], [0.5, 0.7, 1.65]],
        1:[[2, 0.6, 1.2]],
        2:[[4, 1.6, 1.4], [4.6, 1.7, 1.45], [4.5, 1.8, 1.6], [4.7, 1.8, 1.75], [4.9, 1.9, 1.9],
            [7, 2, 2.5], [12, 2.5, 3], [13, 2.5, 4.2], [18, 2.6, 3], [6, 1.7, 1.7],
              [7, 2.5, 3], [9, 2.5, 3], [12, 2.5, 3], [18, 2.5, 3.5]],
        3:[[2, 1, 1.2]]
    }
    Object_ave_height = {
        0:1.7,
        1:1.2,
        2:1.5,
        3:1.2,
        4:3,
        5:2.5
    }

    def __init__(self, ID):
        self.ID = int(ID)
        self.classes = []
        self.Positions = []
        self.Locations = []
        self.BirdEyePos = []
        self.headings = []  
     
        Tracks.all.append(self)

        # Define the Kalman Filter
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # Define the state transition matrix
        self.kf.F = np.array([[1, 0, Tracks.detla_t, 0], [0, 1, 0, Tracks.detla_t], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Define the measurement matrix
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Define the measurement noise covariance matrix
        self.kf.R = np.array([[1, 0], [0, 1]])

        # Define the process noise covariance matrix
        self.kf.Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.kf.Q[2:,2:] *= 0.5

        # Initialize the covariance matrix
        self.kf.P =  1e-1 * np.eye(4, dtype=float)
        self.kf.P[2:,2:] *= 1000.

        self.time_since_update = 0
        self.Initialized = False
        self.age = 1


    def __repr__(self):
        return f"Track {self.ID}"

    def UpdateClass(self):
        self.cls = int(self.Det[5])
        self.classes.append(self.cls)

        if self.classes.count(5) > 5:
            self.cls = 5
        
        if self.classes.count(4) > 5:
            self.cls = 4

    def UpdateStops(self) :
        if len(self.Positions) > 10:
            if np.linalg.norm(self.Position - self.Positions[-10]) < 2:
                if self.motion == "Moving":
                    self.motion = "Stopped"
            else:
                self.motion = "Moving"
    
    def UpdateHeading(self):
        self.UpdateStops()
        if (self.motion == "Moving" or len(self.headings) < 10):
            angle = atan(self.kf.x[3]/self.kf.x[2])
            self.heading = angle if angle >= 0 else angle + pi
        else:
            self.heading = np.array(self.headings[-10:-5]).mean()
        self.headings.append(self.heading)


    def RecordLocation(self):
        self.Positions.append(self.Position)
        Location, BirdEyePos = Transformer.Transform(self.Position)
        self.Locations.append(Location)
        self.BirdEyePos.append(BirdEyePos)
    
    def ComputeLowerSurface(self):
        theta = self.heading 
        A = self.CUBE[0]/2 * np.array([cos(theta), sin(theta)]) 
        B = self.CUBE[1]/2 * np.array([sin(theta), -cos(theta)]) 
        LowerSurface = np.array([self.Position + A + B, self.Position + A - B, self.Position -A - B, self.Position -A + B])
        Points = []
        for point in LowerSurface:
            Points.append(Transformer.Transform(point)[1])
        return np.array(Points)
    
    def Predict(self):
        self.kf.predict()
        self.Position = self.kf.x[0:2]
        self.kf.update(self.Position)
        self.age += 1
        self.time_since_update = self.age

    def Update(self, Det):
        self.Det = Det
        self.UpdateClass()
        self.height = Tracks.Object_ave_height[self.cls]
        self.BBox3D = []
        self.Position = Transformer.defisheye(self.Det[1:3], 'Iterative', self.height)
        if self.Position.size == 0:
            if self.Initialized:
                self.Predict()
        else:
            if self.Initialized:
                self.kf.predict()
                self.UpdateHeading()
                self.BBox3D, self.CUBE, self.type = self.compute_3dBBox()
                #print(f"Track {self.ID} is a {Tracks.Names[self.cls]} with type {self.type}")
                if self.CUBE:
                    Position = Transformer.defisheye(self.Det[1:3], 'Iterative', self.CUBE[2])
                self.Position = Position if Position.size != 0 else self.Position
                self.kf.update(self.Position)
                self.Position = self.kf.x[0:2]
                self.RecordLocation()
                self.age = 0
                
            else:
                    self.Positions.append(self.Position)
                    self.Previous_position = self.Position
                    self.RecordLocation()
                    self.kf.x = np.array([self.Position[0], self.Position[1], 0, 0])
                    self.Initialized = True
                    self.motion = "Moving"
    
    def compute_3dBBox(self):
        def bbox_iou(box2):
            box1 = [self.Det[1] - self.Det[3]/2, self.Det[2] - self.Det[4]/2, self.Det[1] + self.Det[3]/2, self.Det[2] + self.Det[4]/2]
            # Calculate the (x, y) coordinates of the intersection rectangle
            xA = max(box1[0], box2[0])
            yA = max(box1[1], box2[1])
            xB = min(box1[2], box2[2])
            yB = min(box1[3], box2[3])

            # Compute the area of intersection rectangle
            interWidth = max(0, xB - xA)
            interHeight = max(0, yB - yA)
            interArea = interWidth * interHeight
            box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            iou = interArea / float(box1Area + box2Area - interArea)
            return iou
    
        if self.cls in [2, 4, 5]:
            Dimentions = Tracks.ObjectDim[2] 
        else:
            Dimentions = Tracks.ObjectDim[self.cls]
        IOUs = []
        ImgCUBES = []
        for Dim in Dimentions:
            L, W, H = Dim
            Bottom_Center = Transformer.defisheye(self.Det[1:3], 'None-Iterative', H)
            if Bottom_Center.size == 0:
                continue
            theta = self.heading 
            A = L/2 * np.array([cos(theta), sin(theta)]) 
            B = W/2 * np.array([sin(theta), -cos(theta)]) 
            LowerSurface = np.array([Bottom_Center + A + B, Bottom_Center + A - B, Bottom_Center -A - B, Bottom_Center -A + B])
            UpperSurface = LowerSurface /(1 - H / Transformer.H)
            LowerSurface = Transformer.fisheye(LowerSurface)
            UpperSurface = Transformer.fisheye(UpperSurface)
            ImgCUBES.append([LowerSurface, UpperSurface])
            Xmin = min(min(LowerSurface[:, 0]), min(UpperSurface[:, 0]))
            Ymin = min(min(LowerSurface[:, 1]), min(UpperSurface[:, 1]))
            Xmax = max(max(LowerSurface[:, 0]), max(UpperSurface[:, 0]))
            Ymax = max(max(LowerSurface[:, 1]), max(UpperSurface[:, 1]))
            IOUs.append(bbox_iou([Xmin, Ymin, Xmax, Ymax]))
        
        # find the index of the best IOU
        if IOUs:
            CUBE_idx = IOUs.index(max(IOUs))
            return ImgCUBES[CUBE_idx], Dimentions[CUBE_idx], CUBE_idx
        else:
            return [], [], None


           
        

 