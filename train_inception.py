import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import config
from tqdm import tqdm
import time
import os
import shutil
from torch.utils.tensorboard import SummaryWriter

# TensorBoard summary writer
log_folder = 'runs/banana_inception_BS32lr1e-5_splitted_augmented'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

checkpoint_path = log_folder + '/checkpoint.pth'
writer = SummaryWriter(log_folder)

# Hyperparameters
batch_size = 32
learning_rate_start = 0.00001
num_1st_phase_epochs = 0
num_epochs = 100
start_epoch = 0

n=3 #max save last model
n_best = 5
saved_models = []
saved_best_models = []

min_val_loss = 1e9
overfit = False

# Data transforms
# Define image preprocessing function
transform = config.transform_train
transform_val = config.transform_val
# Load datasets
print("loading train data")
train_dataset = datasets.ImageFolder(root=config.dataset_train_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

print("loading validate data")
val_dataset = datasets.ImageFolder(root=config.dataset_validate_path, transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print("Init model")
# Load pre-trained InceptionV3 model
model = config.model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate_start)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=3)

if os.path.exists(checkpoint_path):
    print(f"Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    min_val_loss = checkpoint['min_val_loss']
    saved_models = checkpoint['last_n_model']
    saved_best_models = checkpoint['last_n_best_model']
else:
    print("No checkpoint found, starting from scratch.")
    
# Loss and optimizer
criterion = nn.CrossEntropyLoss()

print("Training")
# Training loop
def train(num_epochs: int, phase: str):
    global min_val_loss, overfit, batch_size, start_epoch, saved_models
    
    for epoch in range(start_epoch,num_epochs):
        print(f"Training epoch {epoch + 1}/{num_epochs} - Phase: {phase}")
        start_time = time.time()  # Start timer for epoch
        
        model.train()
        running_loss = 0.0
        
        # Add tqdm progress bar to the training loop
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}", mininterval=10)):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            if isinstance(outputs, tuple):  # Handle InceptionV3 outputs
                outputs = outputs.logits
            loss = criterion(outputs, labels)  # Use outputs directly for the loss calculation

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, val_labels in tqdm(val_loader, desc=f"Evaluating Epoch {epoch + 1}", mininterval=10):
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                if isinstance(val_outputs, tuple):  # Handle InceptionV3 outputs
                    val_outputs = val_outputs.logits
                val_loss += criterion(val_outputs, val_labels).item()

        # Calculate average losses
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        writer.add_scalar(f'Loss/{phase}/train', avg_train_loss, epoch + 1)
        writer.add_scalar(f'Loss/{phase}/val', avg_val_loss, epoch + 1)

        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar(f'Learning Rate/{phase}', current_lr, epoch + 1)

        epoch_duration = time.time() - start_time
        print(f'Epoch [{epoch + 1}/{num_epochs}], Duration: {epoch_duration:.2f}s, '
              f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # # Save the model at the end of the epoch
        # model_path = f'{log_folder}/model_epoch_{epoch + 1}.pth'
        # torch.save(model.state_dict(), model_path)
        # saved_models.append(model_path)
        # print("Saved model:", model_path)

        # # Remove old models if there are more than n saved
        # if len(saved_models) > n:
        #     oldest_model = saved_models.pop(0)
        #     if os.path.exists(oldest_model):
        #         os.remove(oldest_model)
        #         print(f"Removed old model: {oldest_model}")
        
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            if len(saved_best_models) > n_best :
                oldest_best_model = saved_best_models.pop(0)
                if os.path.exists(oldest_best_model):
                    os.remove(oldest_best_model)
                    print(f"Removed old model: {oldest_best_model}")
            best_model_path = f'{log_folder}/best_at_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), best_model_path)
            saved_best_models.append(best_model_path)
            print(f"Saved best model {best_model_path}")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'min_val_loss': min_val_loss,
            'last_n_model': saved_models,
            'last_n_best_model' : saved_best_models,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
    writer.flush()

for param in model.parameters():
    param.requires_grad = True
for name, child in model.named_children():
    if name == 'Mixed_7c':
        for param in child.parameters():
            param.requires_grad = False
for name, child in model.named_children():
    if name == 'fc':
        for param in child.parameters():
            param.requires_grad = False
        break

train(num_epochs = num_1st_phase_epochs, phase = "train base model")

for param in model.parameters():
    param.requires_grad = False
for name, child in model.named_children():
    if name == 'Mixed_7c':
        for param in child.parameters():
            param.requires_grad = True
for name, child in model.named_children():
    if name == 'fc':
        for param in child.parameters():
            param.requires_grad = True
        break

train(num_epochs = num_epochs, phase = "train fc layer")

print('Finished Training')
writer.close()
# Save the model
# torch.save(model.state_dict(), log_folder+'/final.pth')