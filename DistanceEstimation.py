import cv2 as cv
import numpy as np

import pyttsx3
print("imported\n")
engine = pyttsx3.init()
voices = engine.getProperty('voices')
print("pyttsx3 initialised\n")

# Distance constants
KNOWN_DISTANCE = 45  # INCHES
PERSON_WIDTH = 16  # INCHES
MOBILE_WIDTH = 3.0  # INCHES

# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255),
           (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
# defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights',
                         'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
print("model created\n")
# object detector funciton /method


def object_detector(image):
    print("object detector called\n")
    print(type(image))
    classes, scores, boxes = model.detect(
        image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]

        label = "%s : %f" % (class_names[classid], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)

        if classid == 0:  # person class id
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

        elif classid ==67: # cell phone
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
    
        # adding more classes for distnaces estimation 

        elif classid ==2: # car
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

        elif classid ==15: # cat
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

        elif classid ==16: # dog
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        
        elif classid ==17: # horse
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

        elif classid ==46: # banana
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])

        elif classid ==47: # apple
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
# in that way you can include as many classes you want 

    # returning list containing the object data. 
    return data_list


def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    print("focal length finder called")
    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    print("distance finder called\n")
    return distance

# reading the reference image from dir 
ref_person = cv.imread('ReferenceImages/image14.png')
ref_mobile = cv.imread('ReferenceImages/image4.png')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")
print("reference\n\n")
# finding focal length 
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
cap = cv.VideoCapture(0)

person_voice = 0
person_voice_notes = ""

cell_phone_voice = 0
cell_phone_notes = ""

class Voice:
    def __init__(self, voice_text) -> None:
        self.voice_text = voice_text
        print("voice module called")
    
    def speak(self, i):
        global person_voice
        global cell_phone_voice
        engine.setProperty('voice', voices[i].id)
        engine.say(self.voice_text)
        engine.runAndWait()
        person_voice = 0
        cell_phone_voice = 0
        

while True:
    ret, frame = cap.read()
    data = object_detector(frame)
    for d in data:
        if d[0] =='person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
            if distance <= 40:
                if person_voice != 1:
                    voice_notes = f"There is a person at {int(distance)} inches"
                    person_voice = 1
                    print(voice_notes)
                    sp = Voice(voice_notes)
                    sp.speak(0)


        elif d[0] =='cell phone':
            distance = distance_finder (focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
            if distance <= 40:
                if cell_phone_voice != 1:
                    cell_phone_notes = f"There is a cell phone at {int(distance)} inches"
                    cell_phone_voice = 1
                    print(cell_phone_notes)
                    sp = Voice(cell_phone_notes)
                    sp.speak(1)

        
        elif d[0] =='cat':
            distance = distance_finder (focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
            if distance <= 40:
                if cell_phone_voice != 1:
                    cell_phone_notes = f"There is a cat at {int(distance)} inches"
                    cell_phone_voice = 1
                    print(cell_phone_notes)
                    sp = Voice(cell_phone_notes)
                    sp.speak(1)

        
        elif d[0] =='dog':
            distance = distance_finder (focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
            if distance <= 40:
                if cell_phone_voice != 1:
                    cell_phone_notes = f"There is a dog at {int(distance)} inches"
                    cell_phone_voice = 1
                    print(cell_phone_notes)
                    sp = Voice(cell_phone_notes)
                    sp.speak(0)

        
        elif d[0] =='horse':
            distance = distance_finder (focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
            if distance <= 40:
                if cell_phone_voice != 1:
                    cell_phone_notes = f"There is a horse at {int(distance)} inches"
                    cell_phone_voice = 1
                    print(cell_phone_notes)
                    sp = Voice(cell_phone_notes)
                    sp.speak(0)

        elif d[0] =='apple':
            distance = distance_finder (focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
            if distance <= 40:
                if cell_phone_voice != 1:
                    cell_phone_notes = f"There is an apple at {int(distance)} inches"
                    cell_phone_voice = 1
                    print(cell_phone_notes)
                    sp = Voice(cell_phone_notes)
                    sp.speak(1)


        elif d[0] =='banana':
            distance = distance_finder (focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
            if distance <= 40:
                if cell_phone_voice != 1:
                    cell_phone_notes = f"There is a banana at {int(distance)} inches"
                    cell_phone_voice = 1
                    print(cell_phone_notes)
                    sp = Voice(cell_phone_notes)
                    sp.speak(0)
        
        
        cv.rectangle(frame, (x, y-3), (x+150, y+23),BLACK,-1 )
        cv.putText(frame, f'Dis: {round(distance,2)} inch', (x+5,y+13), FONTS, 0.48, GREEN, 2)

    cv.imshow('frame',frame)
    
    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
cap.release()

