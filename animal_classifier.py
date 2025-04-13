# ========================
# 1. DATA LOADING & CLEANING (NO CHANGES NEEDED)
# ========================
import os
import numpy as np
from PIL import Image

dataset_path = r"C:\Users\mohdq\OneDrive\Desktop\internship projects\Animal_classi.pro\Animal Classification\dataset"
class_names = os.listdir(dataset_path)

print("Classes found:", class_names)
print("\nNumber of images per class:")
for class_name in class_names[:3]:
    class_path = os.path.join(dataset_path, class_name)
    num_images = len(os.listdir(class_path))
    print(f"{class_name}: {num_images} images")

# Corrupt image check (no changes)
corrupt_files = []
for class_name in class_names:
    class_path = os.path.join(dataset_path, class_name)
    for img_name in os.listdir(class_path)[:50]:
        img_path = os.path.join(class_path, img_name)
        try:
            img = Image.open(img_path)
            img.verify()
        except (IOError, SyntaxError) as e:
            print(f"Corrupt file: {img_path} - {e}")
            corrupt_files.append(img_path)
print(f"\nTotal corrupt files: {len(corrupt_files)}")

# ========================
# 2. EDA (NO CHANGES NEEDED)
# ========================
import matplotlib.pyplot as plt

for class_name in class_names[:2]:
    class_path = os.path.join(dataset_path, class_name)
    for img_name in os.listdir(class_path)[:1]:
        img_path = os.path.join(class_path, img_name)
        img = np.array(Image.open(img_path))
        print(f"\nClass: {class_name}")
        print("Image shape:", img.shape)
        plt.imshow(img)
        plt.title(class_name)
        plt.show()

# ========================
# 3. FEATURE ENGINEERING (CRITICAL FIXES)
# ========================
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog, local_binary_pattern
from joblib import Parallel, delayed

# SINGLE VERSION OF THE FUNCTION (remove duplicates)
def extract_features(img_path):
    try:
        img = imread(img_path)
        img = resize(img, (64, 64))
        
        # HOG Features
        hog_feature = hog(img, orientations=6, pixels_per_cell=(16,16),
                        cells_per_block=(1,1), channel_axis=-1)
        
        # LBP Features (fixed)
        gray = (np.mean(img, axis=2) * 255).astype(np.uint8)
        lbp = local_binary_pattern(gray, P=8, R=1)
        lbp_hist = np.histogram(lbp, bins=10, range=(0, 255))[0]
        
        return np.concatenate([hog_feature, lbp_hist]), os.path.basename(os.path.dirname(img_path))
    except Exception as e:
        print(f"Skipped {img_path}: {str(e)}")
        return None

# Feature extraction (with cooling)
print("\nExtracting features...")
results = []
for class_name in class_names:
    class_path = os.path.join(dataset_path, class_name)
    for i, img_name in enumerate(os.listdir(class_path)[:50]):
        if i % 10 == 0:  # Cooling pause
            print(f"Processed {i} images in {class_name}")
        result = extract_features(os.path.join(class_path, img_name))
        if result is not None:
            results.append(result)

# Prepare data
X = np.array([r[0] for r in results])
y = np.array([r[1] for r in results])
print(f"\nLoaded {len(X)} images with classes: {np.unique(y)}")

# ========================
# 4. MODEL TRAINING (CRITICAL FIXES)
# ========================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Label encoding FIRST
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

# Save model
import joblib
joblib.dump({'model': clf, 'encoder': le}, 'animal_classifier.joblib')