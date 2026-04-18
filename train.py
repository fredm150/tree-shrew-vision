import torch                                         # PyTorch for deep learning
import torch.nn as nn                                # Neural network modules
import torch.optim as optim                          # Optimization algorithms
import os                                            # For file handling
import argparse                                      # For command-line argument parsing
import cv2                                           # OpenCV for image processing
import random                                               # For random operations
import wandb                                           # Weights & Biases for experiment tracking
from torch.utils.data import DataLoader, Dataset, Subset    # For creating custom datasets and data loaders
from pytorch_msssim import ssim                             # For Structural Similarity Index (SSIM) evaluation metric
from data.transform import RandomTrans # Data augmentation and tensor conversion



# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model", default ='deep3d_v1.0_640x360_cpu.pt', type=str) # Check for GPU availability (from inference.py logic)
parser.add_argument("--data", default = './data', type=str) # Path to the dataset directory containing 'left_frames' and 'right_frames' subdirectories
parser.add_argument("--epochs", default = 15, type=int) # Number of training epochs
parser.add_argument("--batch_size", default = 1, type=int) # Batch size for training
parser.add_argument("--lr", default = 1e-4, type=float) # Learning rate for the optimizer
parser.add_argument("--num_workers", default = 0, type=int) # Number of worker processes for data loading (set to 0 for Windows compatibility)
parser.add_argument("--val_split", default = 0.15, type=float) # Fraction of data to use for validation
parser.add_argument("--test_split", default = 0.15, type=float) # Fraction of data to use for testing
parser.add_argument("--no_wandb", action="store_true") # Flag to disable Weights & Biases logging if desired
opt = parser.parse_args()

filename = os.path.basename(opt.model)            # Extract the filename from the provided model path
# Check the filename for resolution information and set rescaling parameters accordingly
if '640x360' in filename:
    xRescale = 640
    yRescale = 360
    name = '360p'
elif '1280x720' in filename:
    xRescale = 1280
    yRescale = 720
    name = '720p'
else:
    raise ValueError("Unknown model resolution!")

if not opt.no_wandb:
    wandb.init(
        project="treeshrew-stereo",
        name=f"{name}_lr{opt.lr}_bs{opt.batch_size}",
        config={
            "model": opt.model,
            "epochs": opt.epochs,
            "batch_size": opt.batch_size,
            "learning_rate": opt.lr,
            "val_split": opt.val_split,
            "test_split": opt.test_split,
            "resolution": name,
    }
)




if 'cuda' in opt.model and torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU:", torch.cuda.get_device_name(0))
else:    
    device = torch.device('cpu')
    print("Using CPU")

os.makedirs('./checkpoints', exist_ok=True)  # Create a directory to save checkpoints if it doesn't exist


# The Treeshrew Dataset Class
class TreeshrewDataset(Dataset):
    def __init__(self, data_dir, sequence_length=6, augment = False):
        # Load data here
        self.left_dir = os.path.join(data_dir, 'left_frames')   # Assuming the extracted frames are stored in 'left_frames' and 'right_frames' subdirectories
        self.right_dir = os.path.join(data_dir, 'right_frames') 
        self.left_images = sorted(os.listdir(self.left_dir))    # Sort to ensure matching pairs (frame_0001.jpg in left matches frame_0001.jpg in right)
        self.right_images = sorted(os.listdir(self.right_dir))
        self.sequence_length = sequence_length
        self.half = sequence_length // 2 # Number of frames before and after the current frame to include in the sequence

        assert len(self.left_images) == len(self.right_images), "Mismatch in number of left and right images, ensure there are equal number of frames in both directories."

        self.valid_indices = list(range(self.half, len(self.left_images) - self.half)) # Valid indices for which we can create full sequences

        self.augment = augment
        self.random_trans = RandomTrans()


    def __len__(self):
        # Return the total number of samples (image pairs)
        return len(self.valid_indices)
    

    def _load_frame(self, directory, filename):
        path = os.path.join(directory, filename)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        img = cv2.resize(img, (xRescale, yRescale))  # Resize to the model's expected input size
        img = img.astype('float32') / 255.0           # Normalize pixel values to [0, 1]
        img = img.transpose(2, 0, 1)                 # Change from HWC to CHW format
        return torch.from_numpy(img)                 # Convert to PyTorch tensor


    def __getitem__(self, idx):
        center = self.valid_indices[idx]  # Get the actual index in the original list of images

        params = RandomTrans.get_transform_params(size=(yRescale, xRescale)) if self.augment else None

        frame_indices = range(center - self.half, center - self.half + self.sequence_length)
        frames = []
        for i in frame_indices:
            img = self._load_frame(self.left_dir, self.left_images[i])
            if self.augment:
                img = self.random_trans(img, params)
            frames.append(img)

        left_sequence = torch.cat(frames, dim=0)

        right_tensor = self._load_frame(self.right_dir, self.right_images[center])
        if self.augment:
            right_tensor = self.random_trans(right_tensor, params)  # same params!

        return left_sequence, right_tensor
        
        

# Initialize the dataset and split into training, validation, and test sets

val_full   = TreeshrewDataset(opt.data, augment=False) # Create a dataset instance for the validation and test sets without augmentation (we only augment the training set)
train_full = TreeshrewDataset(opt.data, augment=True) # Create a dataset instance for the training set with augmentation enabled (we will apply random transformations to the training data to improve generalization)

total = len(val_full)         
indices = list(range(total))  # NO shuffle — preserve chronological order

n_test  = int(opt.test_split * total)  
n_val   = int(opt.val_split * total)
n_train = total - n_val - n_test

train_indices = indices[:n_train]
val_indices   = indices[n_train:n_train + n_val]
test_indices  = indices[n_train + n_val:]

train_dataset = Subset(train_full, train_indices)
val_dataset   = Subset(val_full,   val_indices)
test_dataset  = Subset(val_full,   test_indices)


torch.save(test_indices, 'test_indices.pt') # Save the test indices to a file for later use in evaluation (ensures we evaluate on the same samples after training)
print(f"Split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers) # Create a DataLoader for the training set
val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers) # Create a DataLoader for the validation set

model = torch.jit.load(opt.model, map_location=device) # Load the pre-trained model from the specified path and move it to the appropriate device (GPU or CPU)
model.to(device) 



l1_loss = nn.L1Loss().to(device) # Define L1 loss function for pixel-wise differences
optimizer = optim.Adam(model.parameters(), lr=opt.lr) # Initialize the Adam optimizer with the model parameters and specified learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) # Learning rate scheduler to reduce the learning rate by half every 5 epochs

def combined_loss(pred, target):
    loss_l1 = l1_loss(pred, target) # Calculate L1 loss between the predicted and target images
    loss_ssim = 1 - ssim(pred, target, data_range=1.0, size_average=True) # Calculate SSIM loss (1 - SSIM)
    return 0.5 * loss_l1 + 0.5 * loss_ssim # Combine L1 and SSIM losses with weighting


# ------------------------------------ Validation -------------------------------------------------------------

def validate(model, loader):
    model.eval() # Set the model to evaluation mode (disables dropout and batch normalization)
    total_loss, total_ssim, total_psnr = 0, 0, 0

    with torch.no_grad(): # Disable gradient calculation for validation (saves memory and computations)
        for left_img, true_right in loader:
            left_img = left_img.to(device) # Move the input images to the appropriate device
            true_right = true_right.to(device) # Move the target images to the appropriate device

            pred_right = model(left_img) # Get the model's predictions for the right images

            loss = combined_loss(pred_right, true_right) # Calculate the combined loss for the batch
            

            ssim_val = ssim(pred_right, true_right, data_range=1.0, size_average=True).item() # Calculate SSIM score for the batch
            
            mse = torch.mean((pred_right - true_right) ** 2).item() # Calculate Mean Squared Error for PSNR calculation
            psnr_val = 10 * torch.log10(torch.tensor(1.0 / mse)).item() if mse > 0 else float('inf') # Calculate PSNR score for the batch

            total_loss += loss.item()
            total_ssim += ssim_val
            total_psnr += psnr_val

    n = len(loader) # Get the number of batches in the validation set
    return total_loss / n, total_ssim / n, total_psnr / n # Return the average loss, SSIM, and PSNR scores across all batches in the validation set


# -------------------------------- Training Loop -----------------------------------------------------

best_val_ssim = 0.0

for epoch in range(opt.epochs):
    model.train() # Set the model to training mode (enables dropout and batch normalization)
    running_loss = 0.0

    for batch_idx, (left_img, true_right) in enumerate(train_loader):
        left_img = left_img.to(device)
        true_right = true_right.to(device)

        # Forward pass
        pred_right = model(left_img)

        # Compute loss
        loss = combined_loss(pred_right, true_right)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


        running_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{opt.epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            wandb.log({"batch_loss": loss.item()})

    avg_train_loss = running_loss / len(train_loader) # Calculate average training loss for the epoch
    val_loss, val_ssim, val_psnr = validate(model, val_loader) # Validate the model on the validation set and get the average loss, SSIM, and PSNR scores
    scheduler.step() # Step the learning rate scheduler to adjust the learning rate if necessary

    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val SSIM: {val_ssim:.4f} | Val PSNR: {val_psnr:.2f} dB")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}\n")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": val_loss,
        "val_ssim": val_ssim,
        "val_psnr": val_psnr,
        "learning_rate": optimizer.param_groups[0]['lr'],
})


    if (epoch + 1) % 5 == 0:
        model.save(f'checkpoints/checkpoint_epoch_{epoch + 1}_{xRescale}x{yRescale}.pt')
        torch.save({
            'epoch': epoch + 1,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_ssim': val_ssim,
            'val_psnr': val_psnr,
        }, f'checkpoints/checkpoint_epoch_{epoch + 1}_{name}_meta.pth')
        print(f"Checkpoint saved for epoch {epoch + 1}!")

    # Save the best model based on validation SSIM
    if val_ssim > best_val_ssim:
        best_val_ssim = val_ssim
        model.save(f'best_model_{xRescale}x{yRescale}.pt')
        print(f"New best model saved with Val SSIM: {val_ssim:.4f} at epoch {epoch + 1}!")
        wandb.run.summary["best_val_ssim"] = best_val_ssim
        wandb.run.summary["best_epoch"] = epoch + 1


print("Training complete!")
print(f"Best Validation SSIM: {best_val_ssim:.4f}")
print("Run evaluate.py on test_indices.pt to evaluate the best model on the test set.")
wandb.finish()
