
import cv2
from math import tan, atan, exp, pi, sin, asin, cos
from Positioning.Calibration import Transformer
from filterpy.kalman import KalmanFilter
from Positioning.Kalman import CTRVKalmanFilter
import numpy as np

class Tracks:
    detla_t = 0.1
    all = []
    Names = {0:'Person', 1:'Bike', 2:'Car', 3:'Motorcycle', 4:'Bus', 5:'Truck'}
    Object_ave_height = {
        0:1.75,
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
     
        Tracks.all.append(self)
        self.kf = CTRVKalmanFilter(Tracks.detla_t)
        self.time_since_update = 0
        self.Initialized = False
        self.age = 5

    def __repr__(self):
        return f"Track {self.ID}"

    def UpdateClass(self):
        self.cls = int(self.Det[5])
        self.classes.append(self.cls)

        if self.classes.count(5) > 5:
            self.cls = 5
        
        if self.classes.count(4) > 5:
            self.cls = 4

    def RecordLocation(self):
        self.Positions.append(self.Position)
        Location, BirdEyePos = Transformer.Transform(self.Position)
        self.Locations.append(Location)
        self.BirdEyePos.append(BirdEyePos)
    
    
    def Predict(self):
        self.kf.predict()
        self.Position = self.kf.x[0:2]
        self.kf.update(self.Position)
        self.age += 1
        self.RecordLocation()

    def Update(self, Det):
        self.Det = Det
        self.UpdateClass()
        self.height = Tracks.Object_ave_height[self.cls]
        self.Position = Transformer.defisheye(self.Det[1:3], 'Iterative', self.height)
        if self.Position.size == 0:
            if self.Initialized:
                self.Predict()
        else:
            if self.Initialized:
                self.kf.predict()
                self.kf.update(self.Position)
                self.Position = self.kf.x[0:2]
                self.RecordLocation()
                self.age = 0
                
            else:
                    self.RecordLocation()
                    self.kf.x = np.array([self.Position[0], self.Position[1], 1, 0, 0.0001])
                    self.Initialized = True
                    self.motion = "Moving"
                    self.age = 0
    
           
        

 