# generate_mock_data.py
import cv2
import numpy as np
import os

N_FRAMES = 50  # enough to get a real train/val/test split
os.makedirs('./mock_data/left_frames', exist_ok=True)
os.makedirs('./mock_data/right_frames', exist_ok=True)

for i in range(N_FRAMES):
    # Random noise image — just needs to be a valid image
    left  = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    right = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    
    cv2.imwrite(f'./mock_data/left_frames/frame_{i:04d}.jpg', left)
    cv2.imwrite(f'./mock_data/right_frames/frame_{i:04d}.jpg', right)

print(f"Generated {N_FRAMES} mock frame pairs.")