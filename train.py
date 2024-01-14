import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from CustomTransform import ApplyColorMap
from CustomDataset import CustomDataset
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationTrainer

from CustomTrainer import CustomTrainer

# # Define hyperparameters
# batch_size = 16
# epochs = 50
# learning_rate = 0.001

# # Define transforms for image preprocessing
# transform = transforms.Compose([
#     transforms.Grayscale(),
#     ApplyColorMap(),
#     transforms.Resize((640, 480)),
#     transforms.ToTensor()
# ])

# # Load custom dataset
# full_dataset = Dataset('data/images', 'data/annotations', transform=transform)

# # Split dataset into training and validation sets
# train_size = int(0.8 * len(full_dataset))
# val_size = len(full_dataset) - train_size
# train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# # Create data loaders
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# # Initialize model
model = YOLO('yolov8l.pt')

# # Define loss function and optimizer
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Train model
# for epoch in range(epochs):
#     for i, (images, targets) in enumerate(train_loader):
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, targets)
        
#         # Backward pass and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         # Print progress
#         if (i+1) % 10 == 0:
#             print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
#     # Evaluate model on validation set
#     with torch.no_grad():
#         total_loss = 0
#         for images, targets in val_loader:
#             outputs = model(images)
#             loss = criterion(outputs, targets)
#             total_loss += loss.item()
#         avg_loss = total_loss / len(val_loader)
#         print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_loss:.4f}')
    
#     # Save model checkpoint
#     torch.save(model.state_dict(), f'training/{epoch+1}.pt')

trainer = CustomTrainer('default.yaml')
#trainer.train()