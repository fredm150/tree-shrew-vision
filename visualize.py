# visualize.py
import torch
import cv2
import numpy as np
import os
import argparse
from torch.utils.data import Subset, Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model",      default='best_model_360p.pth',        type=str)
parser.add_argument("--base_model", default='deep3d_v1.0_640x360_cpu.pt', type=str)
parser.add_argument("--data",       default='./mock_data',                 type=str)
parser.add_argument("--n_samples",  default=5,                             type=int, help="How many samples to visualize")
parser.add_argument("--output_dir", default='./visualizations',            type=str)
opt = parser.parse_args()

# ── Resolution & Device ──────────────────────────────────────────────────────
filename = os.path.basename(opt.base_model)
if '640x360' in filename:
    xRescale, yRescale = 640, 360
elif '1280x720' in filename:
    xRescale, yRescale = 1280, 720
else:
    raise ValueError("Unknown model resolution.")

device = torch.device('cuda' if ('cuda' in opt.base_model and torch.cuda.is_available()) else 'cpu')
os.makedirs(opt.output_dir, exist_ok=True)

# ── Dataset (same as train/evaluate) ─────────────────────────────────────────
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
        img = cv2.resize(img, (xRescale, yRescale))
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

# ── Load model ───────────────────────────────────────────────────────────────
model = torch.jit.load(opt.base_model, map_location=device)
model.load_state_dict(torch.load(opt.model, map_location=device, weights_only=True))
model.to(device)
model.eval()
print(f"Loaded model from {opt.model}")

# ── Load test indices so we only visualize unseen data ────────────────────────
if not os.path.exists('test_indices.pt'):
    raise FileNotFoundError("test_indices.pt not found. Run train.py first.")

full_dataset = TreeshrewDataset(opt.data)
test_indices = torch.load('test_indices.pt', weights_only=True)
test_dataset = Subset(full_dataset, test_indices)

n_samples = min(opt.n_samples, len(test_dataset))
print(f"Visualizing {n_samples} samples from test set...")

# ── Generate visualizations ───────────────────────────────────────────────────
def tensor_to_cv2(tensor):
    """Convert a [3, H, W] float tensor (0-1) to a BGR uint8 image for OpenCV"""
    img = tensor.cpu().detach().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = img.transpose(1, 2, 0)          # CHW → HWC
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

with torch.no_grad():
    for i in range(n_samples):
        left_sequence, true_right = test_dataset[i]

        # Run through model
        input_tensor = left_sequence.unsqueeze(0).to(device)   # add batch dim → [1, 18, H, W]
        pred_right   = model(input_tensor).squeeze(0)           # remove batch dim → [3, H, W]

        # The center left frame for display (channels 9-12 out of 18, i.e. the 4th frame)
        center_left = left_sequence[9:12]  

        # Convert all three to OpenCV images
        left_img  = tensor_to_cv2(center_left)
        true_img  = tensor_to_cv2(true_right)
        pred_img  = tensor_to_cv2(pred_right)

        # Add labels
        def add_label(img, text):
            img = img.copy()
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (255, 255, 255), 2, cv2.LINE_AA)
            return img

        left_img = add_label(left_img, "Left Input")
        true_img = add_label(true_img, "Ground Truth Right")
        pred_img = add_label(pred_img, "Predicted Right")

        # Stack side by side
        comparison = np.concatenate([left_img, true_img, pred_img], axis=1)

        out_path = os.path.join(opt.output_dir, f"sample_{i+1:03d}.jpg")
        cv2.imwrite(out_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  Saved {out_path}")

print(f"\nDone. Open the '{opt.output_dir}' folder to see results.")
