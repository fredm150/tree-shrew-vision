import torch
import torch.nn as nn
import os
import argparse
import cv2
from torch.utils.data import DataLoader, Subset, Dataset
from pytorch_msssim import ssim



# Reuse same setup from train.py
parser = argparse.ArgumentParser()
parser.add_argument("--model", default = 'best_model_360p.pth', type = str)
parser.add_argument("--base_model", default='deep3d_v1.0_640x360_cpu.pt',    type=str)
parser.add_argument("--data",       default='./data',                         type=str)
parser.add_argument("--batch_size", default=1,                                type=int)
parser.add_argument("--baseline", action="store_true")
opt = parser.parse_args()

filename = os.path.basename(opt.base_model) # Extract the filename

if '640x360' in filename:
    xRescale = 640
    yRescale = 360
elif '1280x720' in filename:
    xRescale = 1280
    yRescale = 720
else:
    raise ValueError("Unknown model resolution!")

device = torch.device("cuda" if('cuda' in opt.base_model and torch.cuda.is_available()) else 'cpu') # Determine the device to use based on the model filename and GPU availability
print(f"Using device: {device}") # Print the device being used for training (GPU or CPU)


class TreeshrewDataset(Dataset):
    def __init__(self, data_dir, sequence_length=6):
        self.left_dir  = os.path.join(data_dir, 'left_frames')
        self.right_dir = os.path.join(data_dir, 'right_frames')
        self.left_images  = sorted(os.listdir(self.left_dir))
        self.right_images = sorted(os.listdir(self.right_dir))
        self.sequence_length = sequence_length
        self.half = sequence_length // 2

        assert len(self.left_images) == len(self.right_images), \
            "Mismatch between left and right frame counts."

        self.valid_indices = list(range(self.half, len(self.left_images) - self.half))

    def __len__(self):
        return len(self.valid_indices)

    def _load_frame(self, directory, filename):
        path = os.path.join(directory, filename)
        img  = cv2.imread(path)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img  = img.astype('float32') / 255.0
        img  = img.transpose(2, 0, 1)
        return torch.from_numpy(img)

    def __getitem__(self, idx):
        center = self.valid_indices[idx]
        frame_indices = range(center - self.half, center - self.half + self.sequence_length)
        frames = [self._load_frame(self.left_dir, self.left_images[i]) for i in frame_indices]
        left_sequence = torch.cat(frames, dim=0)
        right_tensor = self._load_frame(self.right_dir, self.right_images[center])
        return left_sequence, right_tensor




full_dataset = TreeshrewDataset(opt.data)

if not os.path.exists('test_indices.pt'):
    raise FileNotFoundError("test_indices.pt not found. Make sure you're using the same data split as training.")

test_indices = torch.load('test_indices.pt', weights_only=True) # Load the test indices from the file saved during training (ensures we evaluate on the same samples after training)
test_dataset = Subset(full_dataset, test_indices)
test_loader  = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)
print(f"Test set size: {len(test_dataset)} samples")

# ── Load fine-tuned model ────────────────────────────────────────────────────
model = torch.jit.load(opt.base_model, map_location=device)
if not opt.baseline:
    model.load_state_dict(torch.load(opt.model, map_location=device, weights_only=True))
    print(f"Loaded fine-tuned weights from {opt.model}")
else:
    print("Running baseline evaluation on unmodified model.")
model.to(device)
model.eval()

# ── Evaluate ─────────────────────────────────────────────────────────────────
total_ssim, total_psnr = 0.0, 0.0

with torch.no_grad():
    for left_img, true_right in test_loader:
        left_img   = left_img.to(device)
        true_right = true_right.to(device)

        pred_right = model(left_img)

        ssim_val = ssim(pred_right, true_right, data_range=1.0, size_average=True).item()
        mse      = torch.mean((pred_right - true_right) ** 2).item()
        psnr_val = 10 * torch.log10(torch.tensor(1.0 / mse)).item() if mse > 0 else float('inf')

        total_ssim += ssim_val
        total_psnr += psnr_val

n             = len(test_loader)
final_ssim    = total_ssim / n
final_psnr    = total_psnr / n

print("\n── Final Test Results ──────────────────────────")
print(f"  SSIM: {final_ssim:.4f} ")
print(f"  PSNR: {final_psnr:.2f} dB ")
print("────────────────────────────────────────────────")