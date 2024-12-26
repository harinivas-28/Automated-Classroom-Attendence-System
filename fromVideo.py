import torch
import cv2

# YOLOv5 model (pretrained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load input video (replace with your video path)
video_path = 'test_video.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize a set to track unique people
unique_people = set()

# Loop through each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform object detection
    results = model(frame)
    
    # Convert results to pandas dataframe for easy processing
    detection_results = results.pandas().xyxy[0]
    
    # Filter only 'person' class detections
    person_detections = detection_results[detection_results['name'] == 'person']

    # Loop over each person detected
    for index, row in person_detections.iterrows():
        confidence = row['confidence']
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        # Generate a unique identifier based on bounding box coordinates
        person_id = (x1, y1, x2, y2)

        # Add the unique identifier to the set (automatically handles duplicates)
        unique_people.add(person_id)
        
        # Draw the bounding box and label for each person
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Person {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow('Person Detection', frame)
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print the unique number of people detected
print(f"Unique number of people detected: {len(unique_people)}")

# Optionally, save to a text file
with open("Notebooks/pro/unique_people.txt", 'w') as file:
    file.write(f"Unique number of people detected: {len(unique_people)}\n")
