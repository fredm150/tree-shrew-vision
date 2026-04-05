import cv2
import os

def extract(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame as frame_0001.jpg, frame_0002.jpg, etc.
        file_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(file_path, frame)
        frame_count += 1
        
        # Save the left and right images
    cap.release()
    print(f"Done, extracted {frame_count} frames to {output_folder}")
        


# Extract frames from both videos
extract('left_cam.mp4', './mock_data/left_frames')
extract('right_cam.mp4', './mock_data/right_frames')
