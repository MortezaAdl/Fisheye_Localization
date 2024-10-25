import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from Positioning.sort import *
from Positioning.Tracks import Tracks
import csv


"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, confidences = None, names=None, colors = None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = opt.thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        # conf = confidences[i] if confidences is not None else 0

        color = colors[cat]
        
        if not opt.nobbox:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

        if not opt.nolabel:
            label = str(id) + ":"+ names[cat] if identities is not None else  f'{names[cat]} {confidences[i]:.2f}'
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def Plot3dBBox(K, image, Vehicle_type, color=(0, 0, 0)):
    for point in K[0] :
        image = cv2.circle(image, tuple(point.astype(int)) , radius=2, color=(0, 255, 0), thickness=-1)
    for point in K[1] :
        image = cv2.circle(image, tuple(point.astype(int)) , radius=2, color=(255, 0, 0), thickness=-1)
    # pos in the point in K[0] with minimum minimum x value
    pos = tuple(K[0][np.argmin(K[0][:,0])].astype(int))
    image = cv2.polylines(image,  np.int32([K[0]]), True, color, 1)
    image = cv2.polylines(image,  np.int32([K[1]]), True, color, 1)
    image = cv2.polylines(image,  np.int32([np.vstack([K[0][0:2],K[1][0:2][::-1]])]) , True, color, 1)
    image = cv2.polylines(image,  np.int32([np.vstack([K[0][2:4],K[1][2:4][::-1]])]) , True, color, 1)
    #image = cv2.putText(image, str(Vehicle_type), pos, 0, 0.7, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    return image

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    if not opt.nosave:  
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16


    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names =  ['P', 'B', 'C', 'M','B','T']
    colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255],
              [10, 200, 10], [50, 150, 255]]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    ###################################
    startTime = time.time()
    ###################################
    Classes = [0, 1, 2, 3, 4, 5]

    if opt.ShowBirdeyeView:
        BirdEyeImage = cv2.imread(os.getcwd() + '/Positioning/cfg/Intersection.jpg') 

    frame = 0

    for path, img, im0s, vid_cap in dataset:
        FrameData = []
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            csv_path = str(save_dir / 'labels' / 'Detection')
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                dets_to_sort = np.empty((0,6))
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))

                tracked_dets = sort_tracker.update(dets_to_sort, opt.unique_track_color)
                tracks =sort_tracker.getTrackers()
                for track in tracked_dets:
                    xywh = xyxy2xywh(np.array([track[0:4]]))[0] 
                    FrameData.append([track[8], *xywh.round(), track[4]])

                # draw boxes for visualization
                if len(tracked_dets)>0:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    confidences = None

                    if opt.show_track:
                        #loop over tracks
                        for t, track in enumerate(tracks):
                
                            track_color = colors[int(track.detclass)] if not opt.unique_track_color else sort_tracker.color_list[t]

                            [cv2.line(im0, (int(track.centroidarr[i][0]),
                                            int(track.centroidarr[i][1])), 
                                            (int(track.centroidarr[i+1][0]),
                                            int(track.centroidarr[i+1][1])),
                                            track_color, thickness=opt.thickness) 
                                            for i,_ in  enumerate(track.centroidarr) 
                                                if i < len(track.centroidarr)-1 ] 
                
                #im0 = draw_boxes(im0, bbox_xyxy, identities, categories, confidences, names, colors)

            if opt.DoPos:
                detections_List = np.array(FrameData)
                for track in Tracks.all:
                    track.time_since_update += 1
                    for i in range(len(detections_List)):
                        if (track.ID == detections_List[i][0]):
                            track.time_since_update = 0
                            track.Update(detections_List[i])
                            detections_List = np.delete(detections_List, i, 0) 
                            break
                    if (track.time_since_update > 0):
                        track.BBox3D = []
                        # if (track.age < 5) and len(track.Positions) > 10:
                        #     track.Predict()
                        
                        # else:
                        #     track.age += 5
                #print("Number of new tracks: ", len(detections_List))                                       
                for det in detections_List:
                    track = Tracks(det[0])
                    track.Update(det)
                
                Tracks.all = [track for track in Tracks.all if track.time_since_update < 30] 

                if len(Tracks.Initial_Headings) < 4:
                    for track in Tracks.all:
                        if (len(track.headings) > 15) and track.cls == 2:
                            add = True  
                            for point in Tracks.Initial_Headings:
                                if np.linalg.norm(track.Positions[0] - point[0:2]) < 40:
                                    add = False
                                    break
                            if add:
                                Tracks.Initial_Headings.append(np.array([track.Positions[0][0], track.Positions[0][1], track.kf.x[2], np.array(track.headings[-3:]).mean(), track.kf.x[4]]))
                
                if opt.ShowBirdeyeView:
                    Birdeye_img = BirdEyeImage.copy()

                if opt.BBox3D:
                    for track in Tracks.all:
                        if track.BBox3D:
                            color = colors[track.cls]
                            im0 = Plot3dBBox(track.BBox3D, im0, track.type, color)
                            K = track.ComputeLowerSurface()
                            if opt.ShowBirdeyeView:
                                Birdeye_img = cv2.polylines(Birdeye_img,  np.int32([K]), True, colors[track.cls], 1)
                                cv2.circle(Birdeye_img, track.BirdEyePos[-1] , 2, colors[int(track.cls)], thickness=-1)
                    
                # if opt.ShowBirdeyeView: 
                #     for track in Tracks.all:
                #         if track.age < 5:
                #             cv2.circle(Birdeye_img, track.BirdEyePos[-1] , 2, colors[int(track.cls)], thickness=-1)
                    
                    aspect_ratio = Birdeye_img.shape[1] / Birdeye_img.shape[0]
                    new_width = int(aspect_ratio * im0.shape[0])
                    Birdeye_img = cv2.resize(Birdeye_img, (new_width, im0.shape[0]))
                    im0 = np.concatenate((im0, Birdeye_img), axis=1)
            
            ######################################################
            if save_txt:  # Write to file
                for track in Tracks.all:
                    if track.BBox3D:
                        line_csv = (*track.Det, track.Position[0], track.Position[1], track.Coordinate[0], track.Coordinate[1])
                        with open(csv_path + '.csv', 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(line_csv)
            ######################################################
            # Stream results
            ######################################################
            if opt.show_fps:
                frame += 1
                currentTime = time.time()

                fps = frame/(currentTime - startTime)
                #startTime = currentTime
                
                print(f"FPS: {fps}")
                #cv2.putText(im0, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)

            #######################################################
            if view_img:
                cv2.imshow(str(p), cv2.resize(im0, (1500, 630)))
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = im0.shape[1]
                            h = im0.shape[0]
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    parser.add_argument('--track', action='store_true', help='run tracking')
    parser.add_argument('--show-track', action='store_true', help='show tracked path')
    parser.add_argument('--show-fps', action='store_true', help='show fps')
    parser.add_argument('--thickness', type=int, default=2, help='bounding box and font size thickness')
    parser.add_argument('--seed', type=int, default=1, help='random seed to control bbox colors')
    parser.add_argument('--nobbox', action='store_true', help='don`t show bounding box')
    parser.add_argument('--nolabel', action='store_true', help='don`t show label')
    parser.add_argument('--unique-track-color', action='store_true', help='show each track in unique color')
    parser.add_argument('--DoPos', action='store_true', help='Perform Positioning')
    parser.add_argument('--ShowBirdeyeView', action='store_true', help='Show Birdeye view')
    parser.add_argument('--BBox3D', action='store_true', help='Show 3D Bounding Box')



    opt = parser.parse_args()
    print(opt)
    np.random.seed(opt.seed)

    sort_tracker = Sort(max_age=5,
                       min_hits=2,
                       iou_threshold=0.2) 

    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
