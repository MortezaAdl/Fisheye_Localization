# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:56:16 2022

@author: Morteza_Adl
"""
import sys
import numpy as np
from math import sqrt
from matplotlib import path
import os


def FishEye(P) :
    if len(np.shape(P)) == 1:
        P = np.expand_dims(P, axis = 0)   
    FishEyedPos = np.empty((0,2))
    for point in P:
        Fisheyed = np.int32(Image_radius/sqrt(Calibration**2 + point[0]**2 + point[1]**2)*point)
        Fisheyed[0] = int(Fisheyed[0] + ImgCX)
        Fisheyed[1] = int(ImgCY - Fisheyed[1])
        FishEyedPos = np.vstack([FishEyedPos, Fisheyed])   
    return(FishEyedPos)
                     
def DeFishEye(P):
    # Transfer coordinate origin to the camera position
    if len(np.shape(P)) == 1:
        P = np.expand_dims(P, axis = 0)      
    Q = np.empty(np.shape(P))
    Q[:, 0]= P[:, 0] - ImgCX
    Q[:, 1] = ImgCY  - P[:, 1]
    RecPos = np.empty((0,2))
    for point in Q:
        if np.linalg.norm(point) > Filter_radius:
            RecPos = np.vstack([RecPos, np.array([1000000, 1000000])])
        else:
            RecPos = np.vstack([RecPos, Calibration/sqrt(Image_radius**2 - point[0]**2 - point[1]**2)*point])   
    return(RecPos)
    

def Find_Zone(Point):
    y= 5
    for Zone in Zones:
        if Zone.Contains(np.reshape(Point, (1,2))) and Zone.status == "Active":
            y = Zone.name
    return(y)


def Distance(Point1, Point2):
    return(sqrt((Point1[0] - Point2[0])**2 + (Point1[1] - Point2[1])**2))

def Distance_Filter(Location, Filtering_Distance):
    if Distance(Location, Rec_Intersection_center) > Filtering_Distance:
        Location  = infinity
    return(Location)


class Zone:
    def __init__(self, name, status, coordinates, path):
        self.name = name
        self.status = status
        self.coordinates = coordinates
        self.path = path
     
        
    def Contains(self, Point):
        return(self.path.contains_points(Point))   

    
  
cwd = os.getcwd();
cfg_file = cwd+ '/Counter/cfg/cfg.txt'
with open(cfg_file, 'r') as f:
    lines = f.readlines()

Parameters = dict()
for line in lines:
    line = line.strip()
    if line == '' or line.startswith('#'):
        continue
    key, val = line.split('=')
    Parameters[key.strip()] = val.strip()
   
def Str2int(Coord):
    for i in range(len(Coord)) :
        Coord[i] = int(Coord[i])
    return(Coord)

ImageWidth = int(Parameters["ImageWidth"]) 
ImageHeight = int(Parameters["ImageHeight"]) 
ImgCX = int(Parameters["FisheyeImageCenterX"]) 
ImgCY = int(Parameters["FisheyeImageCenterY"]) 
Image_radius = int(Parameters["ImageRadius"])
Filter_radius = int(float(Parameters["Filter"]) * Image_radius)
Calibration = float(Parameters["Rec_Calibration"])
Intersection_vertices = np.reshape(Str2int(Parameters["Intersection"].split()), (4,2))
ZoneData = [0]*4
for i in range(4) :
    Zdata = dict()
    ZD  = Str2int(Parameters["Zone"+ str(i+1) + "Coordinates"].split())
    Zdata["Coordinates"] = np.reshape(ZD, (int(len(ZD)/2), 2))
    Zdata["Status"]      = Parameters["Zone"+ str(i+1) + "Status"]
    Zdata["Center"]      = np.mean(Zdata["Coordinates"], axis=0)
    try:
        Zdata["ASLN"]    = int(Parameters["Zone"+ str(i+1) + "_Arriving_St_Lane_num"])
        Zdata["LSLN"]    = int(Parameters["Zone"+ str(i+1) + "_Leaving_St_Lane_num"])
        ZoneData[i] = Zdata
    except:
        raise("The lane numbers are not specified in the cfg.txt file")

Int_ver = DeFishEye(Intersection_vertices)
Rec_Intersection_center = np.mean(Int_ver, axis=0)
CamLoc = DeFishEye(np.array([ImgCX, ImgCY]))[0]

for i in range(4) :
    ZoneData[i]["Coordinates"] = DeFishEye(ZoneData[i]["Coordinates"])

    
infinity = np.array([1000000, 1000000])    
           
Zones = [Zone(0,"Active", Int_ver, path.Path(Int_ver))]
for i in range(4):
    Zones.append(       
        Zone(i+1,
             ZoneData[i]["Status"],
             ZoneData[i]["Coordinates"],
             path.Path(ZoneData[i]["Coordinates"])
             )
        )