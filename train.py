import math
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
# from models import Deep3D


# Check for GPU availability (from inference.py logic)
parser = argparse.ArgumentParser()
parser.add_argument("--model", default='deep3d_v1.0_640x360_cpu.pt', type=str)
opt = parser.parse_args()
filename = os.path.basename(opt.model)

# Check the filename for resolution information and set rescaling parameters accordingly
if '640x360' in filename:
    xRescale = 640
    yRescale = 360
elif '1280x720' in filename:
    xRescale = 1280
    yRescale = 720
else:
    raise ValueError("Unknown model resolution!")

if 'cuda' in opt.model and torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU:", torch.cuda.get_device_name(0))
else:    
    device = torch.device('cpu')
    print("Using CPU")


# The dataset class
class TreeshrewDataset(Dataset):
    def __init__(self, data_dir):
        # Load data here
        self.left_dir = os.path.join(data_dir, 'left_frames')
        self.right_dir = os.path.join(data_dir, 'right_frames')
        self.left_images = sorted(os.listdir(self.left_dir))    # Sort to ensure matching pairs (frame_0001.png in left matches frame_0001.png in right)
        self.right_images = sorted(os.listdir(self.right_dir))

        assert len(self.left_images) == len(self.right_images), "Mismatch in number of left and right images, ensure there are equal number of frames in both directories."


    def __len__(self):
        # Return the total number of samples (image pairs)
        return len(self.left_images)

    def __getitem__(self, idx):
        left_path = os.path.join(self.left_dir, self.left_images[idx])
        right_path = os.path.join(self.right_dir, self.right_images[idx])
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)

        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        left_img = cv2.resize(left_img, (xRescale, yRescale))
        right_img = cv2.resize(right_img, (xRescale, yRescale))

        left_img = left_img.astype('float32') / 255.0   
        right_img = right_img.astype('float32') / 255.0

        left_img = left_img.transpose(2, 0, 1)   
        right_img = right_img.transpose(2, 0, 1)

        left_tensor = torch.from_numpy(left_img)
        right_tensor = torch.from_numpy(right_img)

        # --- THE DEEP3D TEMPORAL HACKS ---
        # Stack the image 6 times to mimic the (x1, x2, x0, x3, x4, x5) video context
        # This converts the [3, H, W] tensor into an [18, H, W] tensor
        left_sequence = torch.cat([left_tensor] * 6, dim=0)

        return left_sequence, right_tensor
        


# The Setup
# initilize the dataset and dataloader


# Initialize the model, loss function, and optimizer
model = torch.jit.load(opt.model, map_location=device)
model.to(device)
model.train()  # Set the model to training mode

criterion = nn.L1Loss().to(device)  # Using L1 Loss for depth estimation
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
train_loader = DataLoader(TreeshrewDataset('./mock_data'), batch_size=1, shuffle=True)
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (left_img, true_right_img) in enumerate(train_loader):
        left_img = left_img.to(device)
        true_right_img = true_right_img.to(device)

        # Forward pass
        pred_right_img = model(left_img)

        # Compute loss
        loss = criterion(pred_right_img, true_right_img)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), 'treeshrew_fine_tuned.pth')
print("Model saved!")
