import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from PIL import Image
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt

class PeopleCountDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations
            transform (callable, optional): Optional transform to be applied on images
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.target_size = (1223, 373)  # Fixed size for all images
        
        # Verify all image files exist
        missing_files = []
        for idx, row in self.data.iterrows():
            if not os.path.exists(row['filename']):
                missing_files.append(row['filename'])
        
        if missing_files:
            print(f"Warning: {len(missing_files)} image files not found:")
            for file in missing_files[:5]:
                print(f"- {file}")

    def __len__(self):
        return len(self.data)

    def preprocess_image(self, image):
        """Resize image to target size"""
        return cv2.resize(image, (self.target_size[0], self.target_size[1]), interpolation=cv2.INTER_AREA)

    def split_image(self, img):
        """Split the image into two equal halves"""
        width = img.shape[1]
        # Ensure even split
        mid_point = width // 2
        # Make sure both halves have exactly the same width
        img1 = img[:, :mid_point].copy()
        img2 = img[:, mid_point:].copy()
        
        # Double check dimensions are equal
        if img1.shape != img2.shape:
            # Resize second image to match first if needed
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        return img1, img2

    def __getitem__(self, idx):
        # Get image path from CSV
        img_path = self.data.iloc[idx]['filename']
        
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            img = self.preprocess_image(img)
            
            # Split into two views
            img1, img2 = self.split_image(img)
            
            # Verify shapes match
            assert img1.shape == img2.shape, f"Split images have different shapes: {img1.shape} vs {img2.shape}"
            
            if self.transform:
                # Apply same augmentation to both views
                aug1 = self.transform(image=img1)
                aug2 = self.transform(image=img2)
                img1 = aug1['image']
                img2 = aug2['image']
                
                # Verify transformed shapes match
                assert img1.shape == img2.shape, f"Transformed images have different shapes: {img1.shape} vs {img2.shape}"
                
                # Concatenate along channel dimension
                img_combined = torch.cat([img1, img2], dim=0)
            else:
                # If no transform, normalize and convert to tensor manually
                transform = A.Compose([
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
                aug1 = transform(image=img1)
                aug2 = transform(image=img2)
                img1 = aug1['image']
                img2 = aug2['image']
                img_combined = torch.cat([img1, img2], dim=0)
            
            # Get labels from CSV
            labels = torch.tensor([
                self.data.iloc[idx]['total'],
                self.data.iloc[idx]['duplicates'],
                self.data.iloc[idx]['original']
            ], dtype=torch.float32)
            
            return img_combined, labels
            
        except Exception as e:
            print(f"Error processing image {img_path} at index {idx}: {str(e)}")
            raise e

def test_dataset(dataset):
    """Test if the dataset can load and process images correctly"""
    print(f"Testing dataset with {len(dataset)} images...")
    
    # Test first image
    img_combined, labels = dataset[0]
    print(f"First image tensor shape: {img_combined.shape}")
    print(f"Labels: {labels}")
    
    # Test batch loading
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    for batch_idx, (images, batch_labels) in enumerate(loader):
        print(f"\nBatch {batch_idx}:")
        print(f"Batch image shape: {images.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        break

def get_transform(is_train=True):
    if is_train:
        return A.Compose([
            # Don't include RandomResizedCrop as we want to maintain aspect ratio
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                A.RandomGamma(gamma_limit=(80, 120), p=1)
            ], p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

# Create and test dataset
dataset = PeopleCountDataset(
    csv_file='/kaggle/input/classroom-data/photos/photos/labels.csv',
    transform=get_transform(is_train=True)
)

# Test the dataset
test_dataset(dataset)

class PeopleCounterResNet18(nn.Module):
    def __init__(self):
        super(PeopleCounterResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        
        # Freeze early layers
        for param in list(self.resnet.parameters())[:-4]:
            param.requires_grad = False
        
        # Modify the first conv layer to accept concatenated views (6 channels)
        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize new conv layer with weights from pretrained model
        with torch.no_grad():
            self.resnet.conv1.weight[:, :3] = original_conv.weight
            self.resnet.conv1.weight[:, 3:] = original_conv.weight
            
        # Modified final layers
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 3)  # total, duplicates, original
        )

    def forward(self, x):
        return self.resnet(x)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_val_loss = float('inf')
    patience = 7
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

# Create dataloaders
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset, 
    batch_size=2, 
    shuffle=True, 
    num_workers=4
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=2, 
    shuffle=False, 
    num_workers=4
)

# Print dataset info
print(f"Total images: {len(dataset)}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")

# Test loading a batch
for batch_idx, (images, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx}:")
    print(f"Image shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels}")
    break 

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

model = PeopleCounterResNet18()
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)

# Prediction function
def predict(model, image_tensor):
    """Predicts the output for a single image tensor."""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image_tensor)
    return output.cpu().numpy()

# Visualization function
import matplotlib.pyplot as plt
def visualize_predictions(model, dataset, num_samples=3):
    """Visualizes original images, expected outputs, and predicted outputs."""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        img_combined, labels = dataset[idx]
        predicted = predict(model, img_combined)
        print("Expected: ", labels.numpy())
        print("Predicted: ", predicted.flatten())
        # Prepare the original image
        img1 = img_combined[:3].permute(1, 2, 0).cpu().numpy()  # First half
        img2 = img_combined[3:].permute(1, 2, 0).cpu().numpy()  # Second half
        
        # Undo normalization for visualization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img1 = std * img1 + mean
        img2 = std * img2 + mean
        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)

        # Plotting
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img1)
        ax[0].set_title("First Half of Image")
        ax[0].axis("off")
        
        ax[1].imshow(img2)
        ax[1].set_title("Second Half of Image")
        ax[1].axis("off")
        
        ax[2].bar(["Total", "Duplicates", "Original"], labels.numpy(), alpha=0.6, label="Expected")
        ax[2].bar(["Total", "Duplicates", "Original"], predicted.flatten(), alpha=0.6, label="Predicted")
        ax[2].set_title("Predictions vs Expected")
        ax[2].legend()
        
        plt.show()

checkpoint = torch.load("/kaggle/working/best_model.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.load_state_dict(checkpoint['model_state_dict'])

# Visualize model predictions
visualize_predictions(model, dataset)