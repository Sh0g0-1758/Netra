# Import necessary libraries
import cv2                # OpenCV library for computer vision
import subprocess         # subprocess library to execute command-line processes
import pygame             # pygame library for audio playback

# Define a list to store class names
classNames = []

# Specify the path to the file containing class names (COCO dataset)
classFile = "/home/pi/Desktop/Object_Detection_Files/coco.names"

# Read class names from the specified file
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Define paths to the model configuration and weights files
configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

# Create an instance of the object detection model using OpenCV
net = cv2.dnn_DetectionModel(weightsPath, configPath)

# Set input size, scale, mean, and swap channels for object detection
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Define a function to detect objects in an image
def getObjects(img, thres, nms, draw=True, objects=[]):
    # Perform object detection using the specified image, threshold, and non-maximum suppression parameters
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)

    # If a specific list of objects is not provided, use all class names
    if len(objects) == 0:
        objects = classNames

    objectInfo = []

    # Process detected objects and draw bounding boxes if 'draw' is True
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(
                        img,
                        classNames[classId - 1].upper(),
                        (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        img,
                        str(round(confidence * 100, 2)),
                        (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
    
    # Return image with detected objects and their info 
    return img, objectInfo

# Main part of the code
if __name__ == "__main":

    # Initialize video capture from the default camera (camera index 0)
    cap = cv2.VideoCapture(0)

    # Set video capture dimensions
    cap.set(3, 640)
    cap.set(4, 480)

    # Notify the start of object detection
    text_to_speak = "Starting the Object Detection device, will start detecting objects in your surroundings soon"
    festival_output = subprocess.check_output(["festival", "--tts"], input=text_to_speak.encode())
    pygame.mixer.init()
    audio = pygame.mixer.Sound(buffer=festival_output)
    audio.play()
    pygame.time.wait(int(audio.get_length() * 1000))

    while True:
        # Read a frame from the video feed
        success, img = cap.read()

        # Perform object detection with specified threshold and non-maximum suppression parameters
        result, objectInfo = getObjects(img, 0.45, 0.2)

        # Print detected object information
        print(objectInfo)

        # If objects are detected, provide audio feedback for each object
        if len(objectInfo) > 0:
            for i in objectInfo:
                print(i[-1])
                text_to_speak = "there is a " + i[-1] + " in front of you"
                festival_output = subprocess.check_output(["festival", "--tts"], input=text_to_speak.encode())
                pygame.mixer.init()
                audio = pygame.mixer.Sound(buffer=festival_output)
                audio.play()
                pygame.time.wait(int(audio.get_length() * 1000))

        # Wait for a key press (1 millisecond) and continue processing the next frame
        cv2.waitKey(1)
