# ------------------------------------------------------
#Dog vs Cat Image Classification using SVM (Python)
# ------------------------------------------------------

import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# Set Dataset Paths
# -----------------------------------
cat_folder = r"C:\Users\saini\OneDrive\Documents\AI and ML\Machine learning\Unsupervised_learning\SVM\train\cats"
dog_folder = r"C:\Users\saini\OneDrive\Documents\AI and ML\Machine learning\Unsupervised_learning\SVM\train\dogs"

image_size = 64  # Resize images to 64x64 for simplicity
X = []
y = []

# -----------------------------------
# Load and Preprocess Images
# -----------------------------------
def load_images_from_folder(folder, label):
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (image_size, image_size))
            X.append(img.flatten())
            y.append(label)
        else:
            print(f"Warning: Couldn't read image {img_path}")

print("Loading Cat Images...")
load_images_from_folder(cat_folder, 0)  # Label 0 = Cat

print("Loading Dog Images...")
load_images_from_folder(dog_folder, 1)  # Label 1 = Dog

X = np.array(X)
y = np.array(y)

print(f"Total images loaded: {len(X)}")
print(f"Dataset shape: {X.shape}")

# -----------------------------------
#  Train-Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# -----------------------------------
# Train SVM Classifier
# -----------------------------------
print("Training SVM Model...")
model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(X_train, y_train)

# -----------------------------------
# Model Evaluation
# -----------------------------------
print("\nEvaluating on Test Set...")
y_pred = model.predict(X_test)

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Dog vs Cat Classification')
plt.show()

# -----------------------------------
# Single Image Prediction with Safety Check
# -----------------------------------
def predict_single_image(img_path):
    if not os.path.exists(img_path):
        print(f"Error: File does not exist at {img_path}")
        return "Invalid Path!"

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Cannot load image at {img_path}")
        return "Invalid Image!"
    
    img = cv2.resize(img, (image_size, image_size)).flatten().reshape(1, -1)
    pred = model.predict(img)
    return "Dog " if pred[0] == 1 else "Cat "

# Test with a sample image (Change path accordingly)
test_img_path = r"C:\Users\saini\OneDrive\Documents\AI and ML\Machine learning\Unsupervised_learning\SVM\train\cats\cat.953.jpg"
print(f"\n Prediction for {test_img_path}: ", predict_single_image(test_img_path))

test_imgs_path = r"C:\Users\saini\OneDrive\Documents\AI and ML\Machine learning\Unsupervised_learning\SVM\train\dogs\dog.15.jpg"
def predict_single_image(img_path):
    if not os.path.exists(img_path):
        print(f" Error: File does not exist at {img_path}")
        return "Invalid Path!"

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f" Error: Cannot load image at {img_path}")
        return "Invalid Image!"

    img = cv2.resize(img, (image_size, image_size)).flatten().reshape(1, -1)
    pred = model.predict(img)
    return "Dog " if pred[0] == 1 else "Cat "
print(f"\n Prediction for {test_imgs_path}: ", predict_single_image(test_imgs_path))
