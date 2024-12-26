import torch
import cv2
import numpy as np

# YOLOv5 model (pretrained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Replace with your mobile camera's IP webcam link
url = 'https://192.168.24.57:1234/video'  # Replace with your phone's IP address

# Dictionary to store object names and counts
class_counts = {}

# Access mobile camera feed
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform object detection
    results = model(frame)
    
    # Convert results to numpy array for drawing on the frame
    detection_results = results.pandas().xyxy[0]  # Bounding boxes, classes, and confidence scores
    
    # Set to track detected objects in the current frame
    detected_in_frame = set()

    # Display the output frame with detection info
    for index, row in detection_results.iterrows():
        class_name = row['name']
        confidence = row['confidence']

        # Only count unique objects in the current frame
        if class_name not in detected_in_frame and confidence>0.7:
            detected_in_frame.add(class_name)
            
            # Update the count in the dictionary (only once per frame)
            if class_name in class_counts:
                class_counts[class_name] = max(class_counts[class_name],confidence)
            else:
                class_counts[class_name] = confidence
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        # Draw the bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow('YOLOv5 Object Detection', frame)
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print the final detected object classes and counts
print("Objects are saved in the file")
with open("./objects.txt", 'w') as file:
    for class_name, count in class_counts.items():
        file.write(f"{class_name}: {count}\n")