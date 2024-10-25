# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:22:00 2022

@author: adlm
"""

# importing the module
import os
import cv2
import numpy as np
from math import sqrt, ceil
from tkinter import *
from tkinter import filedialog
  
# function to display the coordinates of
# of the points clicked on the image

        
def Save():
    cv2.imwrite("Counter/cfg/CfgImage.jpg", img)
    def VectorizedString(P):
        string = ''
        Q = np.reshape(np.array(P), -1)
        for number in Q:
            string += str(round(number)) + ' '
        return(string)
    
    with open('Counter/cfg/cfg.txt', 'w') as f:
        f.write('# Configuration parameters for data analytics algorithm\n\n')
        f.write('# Image Data'  +'\n')
        f.write('ImageHeight = ' + str(imgshape[0]) +'\n')
        f.write('ImageWidth = ' + str(imgshape[1]) +'\n')
        f.write('FisheyeImageCenterX = ' + str(ImageCenter[0]) + '\n')
        f.write('FisheyeImageCenterY = ' + str(ImageCenter[1]) + '\n\n')
        f.write('# Defisheye function parameters'  +'\n')
        f.write('ImageRadius = ' + str(Image_radius) + '\n')
        f.write('Rec_Calibration = ' + str(Calibration) +'\n')
        f.write('Filter = ' + str(Filter_radius/Image_radius) +'\n\n')
        f.write('# Zones, intersection, and box coordinates'  +'\n\n')
        f.write('Intersection = ' + VectorizedString(Intersection_Points) +'\n\n')
        for i in range(4) :
            f.write('Zone'+ str(i+1) + 'Coordinates = ' + VectorizedString(ZoneCoord[i]) +'\n')
            f.write('Zone'+ str(i+1) + 'Status = Active ' + '\n')
            f.write('Zone'+ str(i+1) +'_Arriving_St_Lane_num = 0' + '\n')
            f.write('Zone'+ str(i+1) +'_Leaving_St_Lane_num = 0' + '\n\n')
        

def Distance(Point1, Point2):
    return(sqrt((Point1[0] - Point2[0])**2 + (Point1[1] - Point2[1])**2))

def Lines_Intersectoin(Line1, Line2):
    m1 = (Line1[0][1] - Line1[1][1])/ (Line1[0][0] - Line1[1][0])
    m2 = (Line2[0][1] - Line2[1][1])/ (Line2[0][0] - Line2[1][0])
    X  = (Line2[0][1] - Line1[0][1] + m1*Line1[0][0] - m2*Line2[0][0])/(m1 - m2)
    Y  = Line1[0][1] + m1*(X - Line1[0][0])
    return([X, Y])

def CircumCenter(A, B, C): 
    D = 2*(A[0]*(B[1] - C[1]) + B[0]*(C[1] - A[1]) + C[0]*(A[1] - B[1]))
    Cx = ((A[0]**2 + A[1]**2)*(B[1] - C[1]) + (B[0]**2 + B[1]**2)*(C[1] - A[1]) + (C[0]**2 + C[1]**2)*(A[1] - B[1]))/D
    Cy = ((A[0]**2 + A[1]**2)*(C[0] - B[0]) + (B[0]**2 + B[1]**2)*(A[0] - C[0]) + (C[0]**2 + C[1]**2)*(B[0] - A[0]))/D
    R = Distance([Cx, Cy], A)
    return([round(Cx), round(Cy)], round(R))

def DeFishEye(P):
    # Transfer coordinate origin to the camera position
    if len(np.shape(P)) == 1:
        P = np.expand_dims(P, axis = 0)      
    Q = np.empty(np.shape(P))
    Q[:, 0]= P[:, 0] - ImageCenter[0]
    Q[:, 1] = ImageCenter[1]  - P[:, 1]
    RecPos = np.empty((0,2))
    for point in Q:
        if np.linalg.norm(point) > Filter_radius:
            RecPos = np.vstack([RecPos, infinity])
        else:
            RecPos = np.vstack([RecPos, Calibration/sqrt(Image_radius**2 - point[0]**2 - point[1]**2)*point])   
    return(RecPos)

def FishEye(P) :
    if len(np.shape(P)) == 1:
        P = np.expand_dims(P, axis = 0)   
    FishEyedPos = np.empty((0,2))
    for point in P:
        Fisheyed = np.int32(Image_radius/sqrt(Calibration**2 + point[0]**2 + point[1]**2)*point)
        Fisheyed[0] = int(Fisheyed[0] + ImageCenter[0])
        Fisheyed[1] = int(ImageCenter[1] - Fisheyed[1])
        FishEyedPos = np.vstack([FishEyedPos, Fisheyed])   
    return(FishEyedPos)

def Interpolate(Points) :
    InPo = []
    CS1Points= np.roll(Points, -1, axis=0)
    DePoints = DeFishEye(Points)
    DeCS1Points = DeFishEye(CS1Points)
    for i in range(len(Points)):
        D = int(Distance(DePoints[i], DeCS1Points[i])) + 1
        for j in range(D) :
            P = DePoints[i] + j * (DeCS1Points[i] - DePoints[i])/D
            InPo.append(FishEye(P))
    return(np.array(InPo))

def PlotPolygon(Points) :
    global img
    Int_points = Points
    Points = Interpolate(Points)
    zonemask = np.zeros(img.shape,np.uint8)
    #cv2.fillPoly(zonemask, np.int32([Points]), (0, 255, 0))
    cv2.polylines(zonemask, np.int32([Points]), True, (255, 255, 255), thickness=5)
    img =cv2.addWeighted(img, 1, zonemask, 1, 0)
    # for i in range(len(Int_points)) :
    #     img = cv2.circle(img, np.int32(Int_points[i]), radius=3, color=(255, 0, 0), thickness=-1)
  
def DefineIntersection(Points) :
    Int_Point = []
    LinePts = DeFishEye(np.array(Points))
    LinePts = np.reshape(LinePts,(4, 2, 2))
    CS1LinePts= np.roll(LinePts, 1, axis=0)
    for i in range(4) :
        Int_Point.append(Lines_Intersectoin(LinePts[i], CS1LinePts[i]))
    Int_Point = FishEye(np.array(Int_Point))
    return(Int_Point)

def Correct(x, y) :
    Point = [x - ImageCenter[0], y- ImageCenter[1]]
    m = Point[1]/Point[0]
    x1 = Filter_radius/sqrt(1+ m**2)
    x2 = -x1
    y1 = m * x1
    y2 = m * x2 
    if (x1*Point[0] + y1* Point[1] ) > 0:
        return(int(x1) + ImageCenter[0], int(y1) + ImageCenter[1])
    else:
        return(int(x2) + ImageCenter[0], int(y2) + ImageCenter[1])
   
    
def click_event(event, x, y, flags, params):
    global PreImg, img, Points, Intersection_Points, CS1Intersection_Points, ZoneCoord, Filter_radius, ImageCenter, Image_radius, font, BoxCoord
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        ValidPoint = True
        def PlotCircles(img):
            img = cv2.circle(img, ImageCenter, radius=Image_radius, color=(0, 0, 0), thickness= 1)
            img = cv2.circle(img, ImageCenter, radius=Filter_radius, color=(0, 255, 255), thickness= 2)
            
        def PlotZones(img):
            for i in range(4) :
                PlotPolygon(np.array(ZoneCoord[i]))
                
        def Smoothen(ZoneCoord):
            InPo = [ZoneCoord[0], ZoneCoord[1]]
            DePoints = DeFishEye(np.array(ZoneCoord[2:]))
            D = int(Distance(DePoints[0], DePoints[1])/10) 
            for j in range(D) :
                P = DePoints[0] + j * (DePoints[1] - DePoints[0])/D
                P = FishEye(P)[0]
                x, y = Correct(P[0], P[1])
                InPo.append([x, y])
            InPo.append(ZoneCoord[3])
            return(np.array(InPo))
            
    
        if 11<= len(Points) <=14 :
            x, y = Correct(x, y)
        
        Points.append([x, y])
        PreImg.append(img.copy())
        
        if 4 <= len(Points) <= 11:
            if Distance(Points[-1], ImageCenter) > Filter_radius:
                Points = Points[0:-1]
                PreImg = PreImg[0:-1]
                ValidPoint = False
                
        if len(Points) == 16 :
            Points = Points[0:-1]
            PreImg = PreImg[0:-1]
            ValidPoint = False
                
        if len(Points) == 3 :
            img  = PreImg[0].copy()
            for i in range(3) :
                img = cv2.circle(img, Points[i], radius=3, color=(0, 0, 255), thickness=-1)
            ImageCenter, Image_radius = CircumCenter(Points[0], Points[1], Points[2])
            Image_radius = int(0.95*Image_radius)
            Filter_radius = int(min(ImageCenter[0], ImageCenter[1], imgshape[1] - ImageCenter[0], imgshape[0] - ImageCenter[1], 0.99*Image_radius))
            PlotCircles(img)
            cv2.putText(img, "Please select two points on each stop line", (20,20), font, 0.5, (0, 0, 255), 2)
            cv2.putText(img, "Points should be inside the yellow circle", (20,40), font, 0.5, (0, 0, 255), 2)
            cv2.imshow('image', img)
            print("Image raduis =" , Image_radius)
            print("Investigation circle raduis = ", Filter_radius)
            print("Image Center = ", ImageCenter)
            
        elif len(Points) == 11 : 
            img = PreImg[0].copy()
            for i in range(3,11) :
                img = cv2.circle(img, Points[i], radius=3, color=(0, 0, 255), thickness=-1)
            Intersection_Points = DefineIntersection(Points[3:])
            Center = np.int32(np.mean(np.array(Intersection_Points), axis=0))
            cv2.putText(img, "Intersection", Center, font, 0.5, (0, 0, 255), 2)
            CS1Intersection_Points = np.roll(Intersection_Points, -1, axis=0)
            PlotCircles(img)
            PlotPolygon(Intersection_Points)
            cv2.putText(img, "Please select two points for each zone", (20,20), font, 0.5, (255, 255, 255), 2)  
            cv2.putText(img, "Points should be near to yellow circle", (20,40), font, 0.5, (255, 255, 255), 2) 
            cv2.imshow('image', img)
            
        elif 12 < len(Points) < 16 :
            for i in range(3) :
                if (len(Points) == 13 + i)  :
                    ZoneCoord[i] = [CS1Intersection_Points[i], Intersection_Points[i], np.array(Points[11 + i]), np.array(Points[12 + i])]
                    ZoneCoord[i] = Smoothen(ZoneCoord[i])
                    PlotPolygon(np.array(ZoneCoord[i]))
                    Center = np.int32(np.mean(np.array(ZoneCoord[i]), axis=0))
                    cv2.putText(img, "Zone" + str(i+1), Center, font, 0.5, (0, 0, 255), 2)
                    cv2.imshow('image', img)
                    if len(Points) == 15 :
                        ZoneCoord[3] = [CS1Intersection_Points[3], Intersection_Points[3], np.array(Points[14]), np.array(Points[11])] 
                        ZoneCoord[3] = Smoothen(ZoneCoord[3])
                        img = PreImg[0].copy()
                        PlotZones(img)
                        PlotPolygon(Intersection_Points)
                        cv2.putText(img, "Configuration file was saved", (20,20), font, 0.5, (255, 255, 255), 2)
                        cv2.putText(img, "To exit press any key", (20,40), font, 0.5, (255, 255, 255), 2)
                        for i in range(4) :
                            Center = np.int32(np.mean(np.array(ZoneCoord[i]), axis=0))
                            cv2.putText(img, "Zone" + str(i+1), Center, font, 0.5, (0, 0, 255), 2)
                        cv2.imshow('image', img)
                        Save()
                            
        else :
            # displaying the coordinates on the image window
            if ValidPoint:
                img = cv2.circle(img, (x,y), radius=3, color=(0, 0, 255), thickness=-1)
                cv2.imshow('image', img)
 
    # checking for right mouse clicks    
    if event==cv2.EVENT_RBUTTONDOWN:
 
        # displaying the coordinates
        # on the image window
        if len(PreImg) > 1 :
            img = PreImg[-1].copy() 
            Points = Points[0:-1]
            PreImg = PreImg[0:-1]
            cv2.imshow('image', img)
 
# driver function
if __name__=="__main__":
    cwd = os.getcwd();
    root = Tk()
    root.title("File Dialog box")
    # Return the name and location of the file.
    try: 
        root.filename = filedialog.askopenfile(initialdir=cwd, title="Select a sample image of intersection", filetypes=(("png files", "*.jpg"),("all file", "*.*")))
        img = cv2.imread(root.filename.name, 1)  
        imgshape = img.shape
        root.destroy()
        Calibration = 20
        infinity = [1000, 1000]
        font = cv2.FONT_HERSHEY_SIMPLEX
        Points = []
        ZoneCoord = [0]*4
        PreImg = [img.copy()]
        cv2.putText(img, "Please select three points on the image circle", (20,20), font, 0.5, (255, 255, 255), 2)
        cv2.imshow('image', img)
        cv2.resizeWindow("image", imgshape[1], imgshape[0])
        # setting mouse handler for the image
        # and calling the click_event() 
        cv2.setMouseCallback('image', click_event)
        # wait for a key to be pressed to exit
        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()
    except:
        root.destroy()      
    
       

  

   