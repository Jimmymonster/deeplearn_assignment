import torch
from PIL import Image
from torchvision import datasets
import os
import time
import config
import shutil

start_time = time.time()
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = config.model
model.load_state_dict(torch.load('runs/banana_inception_BS32lr1e-5_splitted_augmented/best_at_epoch_10.pth', weights_only=True) )
# model.load_state_dict(torch.load("/home/jim/inceptionv3/runs/2phase_train/best_at_epoch_3.pth", weights_only=True))
model = model.to(device)
model.eval()

#===== load model time======
print("---load model %s seconds ---" % (time.time() - start_time))

# Define image preprocessing function
transform = config.transform_val

# Load training dataset to get the class names
train_dataset = datasets.ImageFolder(root=config.dataset_train_path, transform=transform)
train_classes = train_dataset.classes  # Complete list of class names from training dataset

# Function to load and preprocess an image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    # w, h = image.size
    # if w < 40 or h < 40:
    #     print(f"ignore {image_path} due to low dimension ({w}x{h})")
    #     return None
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Function to make predictions on a batch of images
def predict_batch(folder_path, confidence_threshold=0.5, batch_size=32):
    images = []
    filenames = []
    results = []

    with torch.no_grad():
        # Loop through images in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                image_path = os.path.join(folder_path, filename)
                image = load_image(image_path)
                if image is not None:
                    images.append(image)
                    filenames.append(filename)

                # When the batch is full, process it
                if len(images) == batch_size:
                    batch_results = process_batch(images, filenames, confidence_threshold)
                    results.extend(batch_results)
                    images.clear()
                    filenames.clear()

        # Process any remaining images that didn’t fit exactly into a batch
        if images:
            batch_results = process_batch(images, filenames, confidence_threshold)
            results.extend(batch_results)

    return results

def process_batch(images, filenames, confidence_threshold):
    images = torch.cat(images)  # Stack images into a batch
    outputs = model(images)  # Get model outputs (logits)

    # Apply softmax to get probabilities from logits
    probs = torch.softmax(outputs, dim=1)
    # print(probs)

    # Get the predicted class indices
    _, predicted = torch.max(probs, 1)

    # Create results with class names and confidences
    batch_results = []
    for i in range(len(filenames)):
        pred_class = predicted[i].item()  # Get the predicted class index
        pred_confidence = probs[i, pred_class].item()  # Get confidence for predicted class
        confidences = probs[i].tolist()  # Get probabilities for all classes

        # Determine the label from the training dataset's class names
        if pred_confidence >= confidence_threshold:
            label = train_classes[pred_class]  # Map class index to class name
        else:
            label = 'unknown'

        batch_results.append((filenames[i], {
            'label': label,
            'confidences': confidences
        }))

    return batch_results

# Inference on a folder of images
folder_path = '/home/jim/inceptionv3/test_picture'
# folder_path = '/home/jimProject/inceptionv3/old_data/aia-dataset/aia/test'
# folder_path = config.dataset_test_path

start_time = time.time()
results = predict_batch(folder_path, confidence_threshold=0.5)
print("---predict first time %s seconds ---" % (time.time() - start_time))
print(f"Predicted {len(results)} results.")

# Save images with new filenames
summary_folder = 'test_summary'
if os.path.exists(summary_folder):
    shutil.rmtree(summary_folder)
os.makedirs(summary_folder, exist_ok=True)

for index, (filename, result) in enumerate(results):
    label = result['label']
    confident = result['confidences']
    print(f"{filename} {label} {confident}")
    if label == 'unknown':
        continue
    # Construct the new filename
    new_filename = f"{label}_{index}_{filename}"
    # Path for saving the image
    source_path = os.path.join(folder_path, filename)
    dest_path = os.path.join(summary_folder, new_filename)
    # Copy image to summary folder
    shutil.copy(source_path, dest_path)

print(f"Results saved in {summary_folder}")
