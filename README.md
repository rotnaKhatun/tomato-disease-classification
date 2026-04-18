# 🍅 Tomato Leaf Disease Classification using Deep Learning

This project focuses on building an image classification system to detect tomato leaf diseases using a deep learning model based on **MobileNetV3 (Small)**. The model is trained on a dataset of tomato leaf images and evaluated using standard classification metrics.

---

## 📌 Project Overview

Tomato plants are highly affected by various leaf diseases that can significantly reduce crop yield. Early and accurate detection is crucial for effective treatment.

This project aims to:
- Classify tomato leaf images into different disease categories
- Use transfer learning with MobileNetV3-Small
- Evaluate model performance using accuracy, confusion matrix, and classification report
- Implement training pipeline using PyTorch

---

## 📂 Dataset

The dataset contains labeled images of tomato leaves categorized into multiple classes (healthy and diseased).

### Structure:
```
train/
    ├── class_1/
    ├── class_2/
    └── ...
val/
    ├── class_1/
    ├── class_2/
    └── ...
```

- Source: Kaggle Tomato Leaf Dataset  
- Input Image Size: 224 × 224  
- Format: RGB images  

---

## ⚙️ Technologies Used

- Python 🐍
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- PIL (Pillow)
- Kaggle Notebook Environment

---

## 🧠 Model Architecture

- Base Model: MobileNetV3-Small (pretrained on ImageNet)
- Transfer Learning: Final classifier layer modified for custom classes
- Loss Function: CrossEntropyLoss
- Optimizer: AdamW
- Input Size: 224 × 224 × 3

---

## 🏗️ Project Pipeline

1. Import Libraries  
2. Device Selection (CPU / GPU)  
3. Dataset Preparation  
   - Custom PyTorch Dataset  
   - Train/Validation Split  
   - DataLoader Creation  
4. Model Preparation  
5. Training & Validation  
6. Testing & Evaluation  
   - Classification Report  
   - Confusion Matrix  

---

## 🚀 Training Details

- Batch Size: 32  
- Epochs: Up to 100  
- Learning Rate: 1e-4  
- Optimizer: AdamW  
- No explicit early stopping used (based on code flow)

---

## 📊 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## 📈 Results (Saved Outputs)

The following outputs are included in the repository:

- `accuracyPlot.PNG` → Training vs Validation Accuracy Plot  
- `confution matrix.PNG` → Confusion Matrix Visualization  

---

## 📁 Repository Structure

```
├── README.md
├── accuracyPlot.PNG
├── confution matrix.PNG
├── model_train.py
├── tomato-leaf-disease-classification.ipynb
```

---

## 💡 Key Features

- Custom Dataset class implementation using PyTorch  
- Transfer Learning with MobileNetV3  
- Clean train-validation pipeline  
- Model evaluation using standard metrics
- Early stopping & checkpoint saving  
- Visual performance analysis (plots & confusion matrix)

---

## 🔮 Future Improvements

- Data augmentation for better generalization  
- Try EfficientNet / ResNet architectures  
- Deploy using Flask or Streamlit  
- Convert model to ONNX for deployment  

--- 
