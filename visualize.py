import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the training history
with open('training_history.pkl', 'rb') as file:
    history = pickle.load(file)

# Set up the plot style
plt.style.use('seaborn')

# 1. Training and Validation Accuracy/Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('accuracy_loss_plot.png')
plt.close()

# 2. Learning Rate over Epochs
if 'lr' in history:
    plt.figure(figsize=(10, 5))
    plt.plot(history['lr'])
    plt.title('Learning Rate over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.savefig('learning_rate_plot.png')
    plt.close()

# 3. Accuracy vs Loss Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=history['loss'], y=history['accuracy'], label='Training')
sns.scatterplot(x=history['val_loss'], y=history['val_accuracy'], label='Validation')
plt.title('Accuracy vs Loss')
plt.xlabel('Loss')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_vs_loss_plot.png')
plt.close()

# 4. Training Progress
epochs = range(1, len(history['accuracy']) + 1)
plt.figure(figsize=(12, 6))
plt.plot(epochs, history['accuracy'], 'bo-', label='Training Acc')
plt.plot(epochs, history['val_accuracy'], 'ro-', label='Validation Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_progress_plot.png')
plt.close()

print("Initial visualizations complete. Check the current directory for the saved plots.")

# Load the saved model
model = load_model('brain_tumor_vgg19_model.h5')

# Set up the test data generator
test_dir = 'tumor_dataset/Testing'
IMAGE_SIZE = (224, 224)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important: keep the order of images
)

# Generate predictions
y_pred = model.predict(test_generator)
y_true = test_generator.classes

# 5. Confusion Matrix
cm = confusion_matrix(y_true, y_pred.argmax(axis=1))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.close()

print("Confusion matrix saved as 'confusion_matrix.png'")

# 6. Classification Report
class_names = list(test_generator.class_indices.keys())
print("\nClassification Report:")
print(classification_report(y_true, y_pred.argmax(axis=1), target_names=class_names))

# 7. ROC Curve and AUC (for multi-class)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

n_classes = len(class_names)
y_test_bin = label_binarize(y_true, classes=range(n_classes))

plt.figure(figsize=(10, 8))
lw = 2
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])

for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=lw,
             label=f'ROC curve of class {class_names[i]} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()

print("ROC curve saved as 'roc_curve.png'")

# 8. Precision-Recall Curve
from sklearn.metrics import precision_recall_curve, average_precision_score

plt.figure(figsize=(10, 8))

for i, color in zip(range(n_classes), colors):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred[:, i])
    average_precision = average_precision_score(y_test_bin[:, i], y_pred[:, i])
    plt.plot(recall, precision, color=color, lw=lw,
             label=f'Precision-Recall curve of class {class_names[i]} '
                   f'(AP = {average_precision:.2f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig('precision_recall_curve.png')
plt.close()

print("Precision-Recall curve saved as 'precision_recall_curve.png'")

print("All visualizations complete. Check the current directory for the saved plots.")