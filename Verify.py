import cv2
import numpy as np
import tensorflow as tf

# Specify your custom model path 
model_path = 'MyObjectDetectionModel/ssd_mobilenet_v2_coco_2018_03_29/saved_model'

# Load your custom MobileNet SSD model 
model = tf.saved_model.load(model_path)

# Get the inference function 
inference_fn = model.signatures['serving_default']

# Open the video file for reading 
video_path = 'Bro.mp4'
cap = cv2.VideoCapture(video_path)

# Define the codec and create a VideoWriter object to save the output video with H.264 codec 
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter('outfin.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the frame to a format suitable for the model    
    input_image = cv2.resize(frame, (300, 300))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = np.array(input_image, dtype=np.uint8)  # Convert to uint8
    input_image = input_image[np.newaxis, ...]

    # Perform inference on the frame
    detections = inference_fn(tf.convert_to_tensor(input_image, dtype=tf.uint8))  # Convert to uint8

    # Process the detections
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy()

    for i in range(boxes.shape[0]):
        ymin, xmin, ymax, xmax = boxes[i]
        left = int(xmin * frame.shape[1])
        top = int(ymin * frame.shape[0])
        right = int(xmax * frame.shape[1])
        bottom = int(ymax * frame.shape[0])

        # Draw bounding boxes
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Get the confidence score
        score = scores[i]

        # Get the detected class
        detected_class = int(classes[i])

        # Display confidence score and class (replace with your class labels)
        label_text = f'Class: {detected_class}, Confidence: {score:.2f}'
        cv2.putText(frame, label_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame with bounding boxes
    cv2.imshow('Object Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
