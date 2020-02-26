import sys
import cv2
import numpy as np
from argparse import ArgumentParser, SUPPRESS

#from openvino.inference_engine import IECore

#from detector import Detector
from helper import *

from imutils.video import FPS
from PIL import Image

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


CLASSES = ["background", "vehicle", "person"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

VERTS = [[ 699,443],[ 587,443],[ 211, 720], [1150,720]]

if not cv2.useOptimized():
    cv2.setUseOptimized(True)
    
    
class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx += 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        try:
            self.file_name = int(file_name[0])
        except:
            self.file_name = file_name[0]


    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()


def combine_hsl_with_original(img, h_y, h_w):
    mask_ = cv2.bitwise_or(h_y,h_w)
    return cv2.bitwise_and(img,img, mask=mask_)


def filter_img_hsv(img):
    hls_img = to_hsv(img)
    yellow_mask = isolate_yellow_hls(hls_img)
    white_mask = isolate_white_hls(hls_img)
    return combine_hsl_with_original(img,yellow_mask,white_mask)


def filter_img_hsl(img):
    hls_img = to_hls(img)
    yellow_mask = isolate_yellow_hls(hls_img)
    white_mask = isolate_white_hls(hls_img)
    return combine_hsl_with_original(img,yellow_mask,white_mask)

def detect_lines(img, debug=False):
    global VERTS
    ysize, xsize = img.shape[0], img.shape[1]
    
    #hls_img = filter_img_hsv(img)
    #gray = grayscale(hls_img)
    #cv2.imshow(' HLS ', hls_img)
    
    
    
    blur_gray = gaussian_blur(grayscale(img), kernel_size=5)
    
    #cv2.imshow(' gray ', blur_gray)
    
    ht = 150  # First detect gradients above. Then keep between low and high if connected to high
    lt = ht//3  # Leave out gradients below
    canny_edges = canny(blur_gray, low_threshold=lt, high_threshold=ht)
    if debug: save_img(canny_edges, 'canny_edges_{0}'.format(index))
    
    #cv2.imshow('canny', canny_edges)
    # Our region of interest will be dynamically decided on a per-image basis 
    regioned_edges, region_lines = region_of_interest(canny_edges)
    

    #cv2.circle(frame,(p1,p2), 5, (0,255,0),-1)
    #print ()
    #print (line_info[0][0]*p1+ p2 + line_info[0][1])
    
    rho = 2
    theta = 3*np.pi/180
    min_line_length = xsize//16
    max_line_gap = min_line_length//2
    threshold = min_line_length//4
    lines, VERTS = hough_lines(regioned_edges, rho, theta, threshold, min_line_length, max_line_gap)
    
    #print (VERTS)

    # Let's combine the hough-lines with the canny_edges to see how we did
    
    overlayed_lines = weighted_img(img, lines)
    # overlayed_lines = weighted_img(weighted_img(img, region_lines, a=1), lines)
    if debug: save_img(overlayed_lines, 'overlayed_lines_{0}'.format(index))
    
    return overlayed_lines
#cv2.imshow('warning', w_img)

def insertWarning(img):
    s_h, s_w, _ = img.shape
    #print (s_w, s_h)
    img = Image.fromarray(img)
    img.paste(w_img, (s_w-w-10, 10)) 
    return np.array(img)

def issue_warning(x1,y1):
    global VERTS
    point = Point(x1, y1)
    polygon = Polygon(VERTS)
    return polygon.contains(point)

def viz(frame,out):
    img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    
    total_vehicle = 0
    warning = False
    for detection in np.array(out).reshape(-1,7):
        idx = int(detection[1])
        confidence = float(detection[2])
        
        xmin = int(detection[3]*frame.shape[1])
        ymin = int(detection[4]*frame.shape[0])
        xmax = int(detection[5]*frame.shape[1])
        ymax = int(detection[6]*frame.shape[0])
        
        if(confidence>0.3):
            label = "{}:{:.0f}%".format(CLASSES[idx], confidence * 100)
            y = ymin - 15 if ymin - 15 > 15 else ymin + 15
            cv2.putText(frame, label, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.rectangle(img, (xmin,ymin),(xmax,ymax), (0,0,255),-1)
            total_vehicle+=1
            
            cv2.circle(frame,(int((xmin+xmax)*0.5),int(ymax)), 5, (0,0,255),-1)
            
            if(warning==False):
                warning = issue_warning((xmin+xmax)/2,ymax)
            #warning = issue_warning(600,700)
            

            #print (warning)
        
    frame = weighted_img(frame,img,a=1.0, b=.5, l=0.)
    
    if(warning):
        frame = insertWarning(frame)
        
    return frame, total_vehicle

w_img = cv2.imread('images/w_img.jpg',1)
h, w = int(w_img.shape[0]*0.3), int(w_img.shape[1]*0.3)
w_img = cv2.resize(w_img, (w,h),interpolation= cv2.INTER_AREA)
w_img = Image.fromarray(w_img)




def main():
    model_xml = "models/vehicle-detection-adas-0002_f16.xml"
    model_bin = "models/vehicle-detection-adas-0002_f16.bin"

    #ie = IECore()
    #detector = Detector(ie, model_xml, model_bin, 0.4, "MYRIAD")
    
    count = 4734
    count_max = 5269
    
    fps = FPS().start()
    
    while (True):
        
        file = 'frame_%d.jpg' %count
        in_filename = 'data/'+ file
        
        #print ('readling file : ', filename)
        frame = cv2.imread(in_filename,1)
 
        count+=1
        if(count>=count_max): break
        
        if (frame is None): continue 

        frame = detect_lines(frame)
        out = detector.detect(frame)        
        frame, t_vehicle = viz(frame,out)
        
        
        
        cv2.putText(frame, 'summary: {:.1f} FPS'.format(float(1 / (detector.infer_time * len(out)))), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200))
        cv2.putText(frame, 'Total Vehicle Detected: {:.0f}'.format(t_vehicle), (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200))
        
        
        cv2.imshow('Vechile ADAS System', frame)
        
        #out_filename = 'output/'+ file
        #cv2.imwrite(out_filename,frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
        
        
        fps.update()
        
        
        
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if __name__ == "__main__":
    sys.exit(main() or 0)

