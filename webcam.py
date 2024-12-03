import cv2
import numpy as np

# Replace with your mobile camera's IP webcam link (if using IP Webcam)
url = 'https://100.75.135.171:8080/video'  # Replace with your phone's IP address
# YOLO model files (adjust paths as needed)
config_path = 'Notebooks/pro/yolov3.cfg'
weights_path = 'Notebooks/pro/yolov3.weights'
names_path = 'Notebooks/pro/coco.names'

# Load YOLO model and classes
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
classes = []
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize variables to store detected classes and counts
detected_classes = set()
class_counts = {}

# Access mobile camera feed
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get frame dimensions
    (h, w) = frame.shape[:2]
    
    # Prepare the frame for object detection
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Perform object detection
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)
    
    # Loop over detections and count objects
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter out weak detections (adjust as needed)
            if confidence > 0.5:
                class_name = classes[class_id]
                
                # Update count for detected class
                if class_name not in detected_classes:
                    detected_classes.add(class_name)
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    else:
                        class_counts[class_name] = 1
    
    # Display the output frame with detection info (optional)
    for class_name, count in class_counts.items():
        cv2.putText(frame, f'{class_name}: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow('Object Detection', frame)
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print final detected classes and counts
print("Detected Object Classes and Counts:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")
