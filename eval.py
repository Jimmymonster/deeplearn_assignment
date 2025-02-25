import os
import shutil
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import cycle
from sklearn.preprocessing import label_binarize
import config

# Hyperparameters
batch_size = 32
eval_folder = config.dataset_test_path
eval_output_folder = "eval_graph"
classes_per_subplot = 10  # Define how many classes to plot per subplot

# Define image preprocessing function
transform = config.transform_val

# Load training dataset to get the complete list of classes
train_dataset = datasets.ImageFolder(root=config.dataset_train_path, transform=transform)
train_classes = train_dataset.classes  # Complete list of classes from training

# Load evaluation dataset
val_dataset = datasets.ImageFolder(root=eval_folder, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Get the list of classes from the evaluation dataset
eval_classes = val_dataset.classes

# Create a mapping from evaluation class index to training class index
eval_to_train_mapping = {val_dataset.class_to_idx[cls]: train_dataset.class_to_idx[cls] for cls in eval_classes if cls in train_classes}

# Load the trained model
model = config.model
# model.load_state_dict(torch.load('best.pth'))
model.load_state_dict(torch.load("runs/banana_inception_BS32lr1e-5_splitted_augmented/best_at_epoch_10.pth", weights_only=True))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Clear eval output folder
if os.path.exists(eval_output_folder):
    shutil.rmtree(eval_output_folder)
os.makedirs(eval_output_folder, exist_ok=True)

# Function to evaluate the model
def evaluate_model(model, dataloader):
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            scores = torch.softmax(outputs, dim=1)

            # Map the evaluation labels and predictions back to the training indices
            labels_mapped = [eval_to_train_mapping[label.item()] for label in labels]
            y_true.extend(labels_mapped)
            y_pred.extend([eval_to_train_mapping[pred.item()] for pred in predicted])
            y_scores.extend(scores.cpu().numpy())

    return y_true, y_pred, y_scores

# Evaluate the model
y_true, y_pred, y_scores = evaluate_model(model, val_loader)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy:.4f}')

report = classification_report(y_true, y_pred, labels=range(len(train_classes)), target_names=train_classes, zero_division=0)
print('Classification Report:')
print(report)

# Save classification report as a text file
report_file_path = os.path.join(eval_output_folder, 'classification_report.txt')
with open(report_file_path, 'w') as f:
    f.write(report)

# Generate a full classification report including all classes
full_report = classification_report(
    y_true, 
    y_pred, 
    labels=range(len(train_classes)), 
    target_names=train_classes, 
    output_dict=True,  # Output as a dictionary for easier processing
    zero_division=0
)
# Filter the report to only include classes with an f1-score less than 1
filtered_report = {
    class_name: metrics 
    for class_name, metrics in full_report.items() 
    if isinstance(metrics, dict) and metrics.get('f1-score', 1.0) < 1.00
}

# Check if filtered_report is empty
filtered_report_str = ""
if not filtered_report:
    print("No classes with f1-score less than 1.0 found.")
else:
    # Prepare the filtered class names and labels
    filtered_labels = [i for i, name in enumerate(train_classes) if name in filtered_report]
    filtered_target_names = [name for name in filtered_report]

    # Check that there are classes to report
    if not filtered_target_names:
        print("No matching target names found for classification report.")
    else:
        # Generate the filtered classification report
        filtered_report_str = classification_report(
            y_true, 
            y_pred, 
            labels=filtered_labels,
            target_names=filtered_target_names,
            zero_division=0
        )

        print(filtered_report_str)
# print('Classification Report (f1-score < 1.00):')
# print(filtered_report_str)
# Save the filtered classification report as a text file
filtered_report_file_path = os.path.join(eval_output_folder, 'misclassification_report.txt')
with open(filtered_report_file_path, 'w') as f:
    f.write(filtered_report_str)

# List to store misclassified information
misclassified_data = []
# Loop through all predictions and find misclassifications
for i, (true_label, pred_label, score) in enumerate(zip(y_true, y_pred, y_scores)):
    if true_label != pred_label:  # If the prediction is wrong
        misclassified_data.append({
            'Image Index': i,  # This can also be image_paths[i] if you're using image file paths
            'True Label': train_classes[true_label], 
            'Predicted Label': train_classes[pred_label],
            'Confidence': score[pred_label]  # Confidence for the predicted label
        })
# Define the header and formatting for the table
header = f"{'Image Index':<15}{'True Label':<30}{'Predicted Label':<30}{'Confidence Level':<20}\n"
separator = '-' * 75 + '\n'
# Write the misclassification report as a text file
misclassified_report_file_path = os.path.join(eval_output_folder, 'misclassified_report_with_confidence.txt')
with open(misclassified_report_file_path, 'w') as f:
    # Write header and separator
    f.write(header)
    f.write(separator)
    # Write each misclassified entry in a table format
    for entry in misclassified_data:
        f.write(f"{entry['Image Index']:<15}{entry['True Label']:<30}{entry['Predicted Label']:<30}{entry['Confidence']:<20.2f}\n")
print(f"Misclassification report saved to {misclassified_report_file_path}")

# Compute confusion matrix (including missing classes)
conf_matrix = confusion_matrix(y_true, y_pred, labels=range(len(train_classes)))
# Plot confusion matrix as a heatmap
plt.figure(figsize=(32, 28))  # Adjust size for better visibility
# Use a mask to hide some values if necessary
mask = np.zeros_like(conf_matrix, dtype=bool)
# Only show the most significant values
for i in range(len(train_classes)):
    for j in range(len(train_classes)):
        if conf_matrix[i, j] < 0:  # Hide cells with low values, adjust threshold as needed
            mask[i, j] = True
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=train_classes, 
            yticklabels=train_classes, mask=mask, cbar_kws={'label': 'Counts'},
            annot_kws={"size": 6})  # Adjust annotation font size
plt.xlabel('Predicted', fontsize=8)
plt.ylabel('True', fontsize=8)
plt.title('Confusion Matrix', fontsize=8)
plt.xticks(rotation=90, fontsize=10)  # Rotate x-axis labels for better readability
plt.yticks(rotation=0, fontsize=10)   # Rotate y-axis labels for better readability
plt.tight_layout()
plt.savefig(os.path.join(eval_output_folder, 'confusion_matrix.png'))
plt.show()

# Compute ROC curve and ROC area for each class
y_true_bin = label_binarize(y_true, classes=range(len(train_classes)))
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(train_classes)):
    if i in np.unique(y_true):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], np.array(y_scores)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    else:
        fpr[i], tpr[i], roc_auc[i] = [0], [0], 0  # Set to 0 for missing classes

# Plot ROC curve in subplots
num_roc_subplots = len(train_classes) // classes_per_subplot + (len(train_classes) % classes_per_subplot != 0)
fig, axes = plt.subplots(num_roc_subplots, 1, figsize=(10, num_roc_subplots * 4))

for subplot_idx in range(num_roc_subplots):
    if num_roc_subplots == 1:
        ax = axes
    else:
        ax = axes[subplot_idx]
    start_idx = subplot_idx * classes_per_subplot
    end_idx = min(start_idx + classes_per_subplot, len(train_classes))

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'pink', 'brown', 'yellow', 'blue'])
    
    for i, color in zip(range(start_idx, end_idx), colors):
        if i in np.unique(y_true):
            ax.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {train_classes[i]} (area = {roc_auc[i]:0.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve for Classes {start_idx + 1} to {end_idx}')
    ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(eval_output_folder, 'roc_curve_subplots.png'))
plt.show()
