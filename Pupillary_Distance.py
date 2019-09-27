#print("Hello word .. .")
import numpy as np
import math, datetime
import dlib, cv2
import os, sys
rom imutils import face_utils
import imutils
from os import listdir
from os.path import join, isfile
# import pandas as pd


# Drawing colors
COLOR_GREEN = (0, 255,0)
COLOR_RED = (0, 0, 255)

# Real width of a credit card in millimeters
CREDIT_CARD_WIDTH_MM = 90

# Use MMOD Convolution Neural Network for face detection
USE_CNN = True


# Show debugging images
DEBUG = False

#Range of valid PD
MINIMUM_PD = 50
MAXIMUM_PD = 78




BASE_PATH = os.path.abspath(os.path.dirname(__file__))

CARD_DETECTOR_PATH = "{base_path}/deps/card_detector.svm".format(base_path=BASE_PATH)
FACE_DETECTOR_PATH = "{base_path}/deps/mmod_human_face_detector.dat".format(base_path=BASE_PATH)
EYE_DETECTOR_PATH = "{base_path}/deps/haarcascade_eye.xml".format(base_path=BASE_PATH)
LANDMARK_PREDICTOR_PATH = "{base_path}/deps/shape_predictor_68_face_landmarks.dat".format(base_path=BASE_PATH)
PUPIL_PREDICTOR_PATH = "{base_path}/deps/pupil_predictor.dat".format(base_path=BASE_PATH)

#This is the Caffe based ResNet+SSD model for face detection to be used in OpenCV
FACE_MODEL_PATH = "{base_path}/deps/res10_300x300_ssd_iter_140000.caffemodel".format(base_path=BASE_PATH)
FACE_PROTO_PATH =  "{base_path}/deps/deploy.prototxt.txt".format(base_path=BASE_PATH)



DLIB_CARD_PREDICTOR_PATH = "{base_path}/deps/card_predictor.dat".format(base_path=BASE_PATH)


dlib_card_predictor = dlib.shape_predictor(DLIB_CARD_PREDICTOR_PATH)
card_detector = dlib.simple_object_detector(CARD_DETECTOR_PATH);
face_detector = dlib.cnn_face_detection_model_v1(FACE_DETECTOR_PATH) if USE_CNN else dlib.get_frontal_face_detector()
eye_detector = cv2.CascadeClassifier(EYE_DETECTOR_PATH);
landmark_predictor = dlib.shape_predictor(LANDMARK_PREDICTOR_PATH)
pupil_predictor = dlib.shape_predictor(PUPIL_PREDICTOR_PATH)


face_detector_cv = cv2.dnn.readNetFromCaffe(FACE_PROTO_PATH, FACE_MODEL_PATH)


#To check that either the PD is in correct  range or not.  . .
def isValidPD(PD):
    PD = round(PD)
    # print(PD, MINIMUM_PD, MAXIMUM_PD)
    # print((PD >= MINIMUM_PD) and (PD <= MAXIMUM_PD))
    return (PD >= MINIMUM_PD) and (PD <= MAXIMUM_PD);


#Calculate thresholded saliency map to isolate black strip of card
#That method has also not been used in the final version of the code ..  .
def getSaliencyMap(pic):    
    (success, saliencyMap) = saliency.computeSaliency(pic)
    saliencyMap = np.uint8(saliencyMap * 255)  
    threshold = cv2.threshold(saliencyMap, 35, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((7, 7), np.uint8)
    erosion = cv2.erode(threshold, kernel, iterations=1)
    return erosion


#That method has been implemented in order find out that either lightening condition 
#is sufficient or not . . . . That method also has not been used in the final version of code . . .


def isLightAdequate(y):
    NUM_BINS = 10;
    total_pixels = y.shape[0] * y.shape[1];
    hist = cv2.calcHist([y],[0],None,[256],[0,256])
    hist = [i[0] for i in hist]
    
    dark_values = hist[0:NUM_BINS];
    light_values = hist[-NUM_BINS:]

    light = round(sum(light_values) / total_pixels * 100.0, 3)
    dark = round(sum(dark_values) / total_pixels * 100.0, 3)

    if abs(dark - light) >= 5:
        return False, "Too dark/Too bright | "
    else:
        return True, ""


#Apply some pre-processing to the image before detecting face in order to adjust the brightness of the image
#That method is not more in use in the code, but still has been developed, just for the testing purpose . .. 
def preProcess(bgr_image):
    print('Applying CLAHE')
    ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
    adequateLight, lightMessage = isLightAdequate(ycrcb[:,:,0])
    ycrcb[:,:,0] = CLAHE.apply(ycrcb[:,:,0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), adequateLight, lightMessage



#Given the face image, try to detect location of pupils
def getPupilLocations(face, bbox, landmarks):

    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY);

    #First try OpenCV's Haar classifier for eye detection
    eyes = eye_detector.detectMultiScale(face_gray);

    left_eye = (0,0);
    right_eye = (0,0);

    if len(eyes) > 1:
        eyes = eyes[0:2]
        (startX, startY, w, h) = eyes[0]
        
        rect = dlib.rectangle(startX, startY, startX + w, startY + h)
        
        #Apply pupil prediction model to determine pupil loaction
        shape = face_utils.shape_to_np(pupil_predictor(face_gray, rect))
        
        fx = 0
        fy = 0
        for (i, (x, y)) in enumerate(shape):
            px = int(x)
            py = int(y)
            fx += int(x + bbox.left())
            fy += int(y + bbox.top())
            cv2.circle(face, (px, py), 1, (0, 0, 255), -1)

        fx = int(fx / 4.0)
        fy = int(fy / 4.0)
        left_eye = (fx,fy)

        (startX, startY, w, h) = eyes[1]
        rect = dlib.rectangle(startX, startY, startX + w, startY + h)
        
        #Apply pupil prediction model to determine pupil loaction
        shape = face_utils.shape_to_np(pupil_predictor(face_gray, rect))

        fx = 0
        fy = 0
        for (i, (x, y)) in enumerate(shape):
            px = int(x)
            py = int(y)
            fx += int(x + bbox.left())
            fy += int(y + bbox.top())
            cv2.circle(face, (px, py), 1, (0, 0, 255), -1)
        fx = int(fx / 4.0)
        fy = int(fy / 4.0)
        right_eye = (fx,fy);

        name = '/home/ubuntu/face_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + '.jpg'
        #cv2.imwrite(name, face)
    else:
        #If Haar cannot detect eyes, then predict eye locations using facial landmarks
        # print('Using Landmarks')
        LEFT_EYE = landmarks[36:42]
        left_x = []
        left_y = []

        for i in LEFT_EYE:
            left_x.append(i[0])
            left_y.append(i[1])
        left_eye_x = round(np.mean(left_x))
        left_eye_y = round(np.mean(left_y))

        # finding landmarks on right eyes
        RIGHT_EYE = landmarks[42:48]
        right_x = []
        right_y = []
        for i in RIGHT_EYE:
            right_x.append(i[0])
            right_y.append(i[1])

        right_eye_x = round(np.mean(right_x))
        right_eye_y = round(np.mean(right_y))

        left_eye = (left_eye_x, left_eye_y)
        right_eye = (right_eye_x, right_eye_y);

    return left_eye, right_eye;

def detectFaces(image):
    # print('Calling OpenCV DNN')
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector_cv.setInput(blob)
    detections = face_detector_cv.forward()

    confidence = detections[0, 0, 0, 2]

    retval = []

    for i in range(0, detections.shape[2]):
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            retval.append(dlib.rectangle(startX, startY, endX, endY))
    return retval


def detect_card_from_face_region(rgb_image, bbox):
    # Detect card in the face region, extending bottom, left, and right by 10%
    width = abs(bbox.right() - bbox.left())
    height = abs(bbox.bottom() - bbox.top())

    new_bottom = bbox.bottom() + int(height * 0.1)
    if new_bottom >= rgb_image.shape[0]:
        new_bottom = rgb_image.shape[0] - 1
    new_left = bbox.left() - int(width * 0.1)
    if new_left < 0:
        new_left = 0
    new_right = bbox.right() + int(width * 0.1)
    if new_right >= rgb_image.shape[1]:
        new_right = rgb_image.shape[1] - 1
    
    #cropped = rgb_image[bbox.top():new_bottom, new_left:new_right, :]
    card_face_rect = dlib.rectangle(new_left, bbox.top(), new_right, new_bottom)

    # (x, y, w, h) = face_utils.rect_to_bb(card_face_rect)
    # cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    card_shape = face_utils.shape_to_np(dlib_card_predictor(rgb_image, card_face_rect))
    return card_shape

def computePupillaryDistance(bgr_image,name):
    
    lightMessage = ''  

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB);
    # rgb_image = bgr_image
    if rgb_image.shape[1] > 500:
        rgb_image = imutils.resize(rgb_image, width=500)

    opencv_detector = True;
    dets = detectFaces(rgb_image)
    
    if not dets:
        opencv_detector = False
        dets = face_detector(rgb_im2age, 1)

    if not dets:
        # print("Here I have reached")
        
        #cv2.putText(photo, 'Face could not be found', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
        return 0, lightMessage + "No Face Found."
    

    det = dets[0]
    if opencv_detector:
        # print('Using OpenCV detections')
        bbox = det
    else:
        bbox = det.rect if USE_CNN else det;
        
    face = bgr_image[bbox.top():bbox.bottom(), bbox.left():bbox.right(), :]

    #cv2.imwrite('/home/ubuntu/face.jpg', face)

    if (face.shape[0] == 0) or (face.shape[1] == 0):
        #cv2.putText(bgr_image, 'Error: Invalid Face', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
        return 0, lightMessage + "Face not completely visible!"

    landmarks = landmark_predictor(rgb_image, bbox)
    landmarks = face_utils.shape_to_np(landmarks)

    left_eye, right_eye = getPupilLocations(face, bbox, landmarks);    
    #cv2.line(bgr_image, (int(left_eye_x), int(left_eye_y)), (int(right_eye_x), int(right_eye_y)), (0, 0, 255), 2)
    
    cv2.line(rgb_image, (int(left_eye[0]), int(left_eye[1])), (int(right_eye[0]), int(right_eye[1])), (0,255,0), 3)
    
    # Finding Eucledian distance between eye balls
    dx = left_eye[0] - right_eye[0]
    dy = left_eye[1] - right_eye[1]

    #Compute PD in pixels
    eyes_distance_in_px = math.sqrt(dx ** 2 + dy ** 2)


    if not isValidPD(MLPD):
        MLPD = 0;
        MLPD_status = lightMessage + "Cannot measure pupillary distance";

    card_points = detect_card_from_face_region(rgb_image, bbox)
    dx = card_points[0][0] - card_points[1][0]
    dy = card_points[0][1] - card_points[1][1]

    card_distance_in_px = math.sqrt(dx ** 2 + dy ** 2)
    # print("pixeles distance")
    # print(card_distance_in_px)

   
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(rgb_image,"Pupillary Distance is" ,(10,500),font,4,(255,255,255),2,cv2.LINE_AA)
    

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA

    PD = (eyes_distance_in_px / card_distance_in_px) * CREDIT_CARD_WIDTH_MM

    cv2.line(rgb_image, (card_points[0][0], card_points[0][1]), (card_points[1][0], card_points[1][1]), (0,255,0), 3)        
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(rgb_image, "Distance: {}mm (Updated Formula)".format( round(PD,2)) ,(10,50), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # cv2.imshow('My_Image . ', rgb_image)
    # cv2.waitKey()
    # exit()

    # cv2.imwrite("/home/hassanahmed/PycharmProjects/PP_TESTING/Pupillary Distance/RESULTS/18_feb_results/" + name + ".jpg",rgb_image)
   
    print(PD)
    if isValidPD(PD):
        #cv2.putText(bgr_image, 'Pupillary Distance Contour = ' + str(round(PD, 1)) + ' mm', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
        return PD, "Ok"
    
    else:
        return " ", "Sorry, Card Detection failed. Try Again!"

    if card_warning_message is not None:
        cv2.putText(bgr_image, card_warning_message, (20, bgr_image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

    #that will be the final output of the compute pujpliaary distance method . .. , PD and the final image on which paterns have been drawn . . .     
    # return PD, rgb_image;


def main_image(path):
    # items = os.listdir(path)

    # path= "/home/hassanahmed/PycharmProjects/PP_TESTING/Pupillary Distance/RESULTS"
    name = os.path.basename(path)

    photo = cv2.imread(path)


    heigt, width ,channel = photo.shape
    # print (heigt, width,channel)

    new_photo = cv2.resize(photo,(3220,4088))

    new_photo = imutils.resize(new_photo, width=  500)

    heigt, width ,channel = new_photo.shape
        
    # A, B =computePupillaryDistance(new_photo,name);
    # print (A)
    # print(B)
    computePupillaryDistance(new_photo,name)

if __name__ == '__main__':

    pic_path=  "/home/hassanahmed/PycharmProjects/PP_TESTING/Pupillary Distance/pic2.jpg"

    # main_image_2(path)
    main_image(pic_path)
