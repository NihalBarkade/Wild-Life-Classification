# 🐾 Wild-Life Classification

A deep learning-based animal classifier using MobileNetV2 trained on 3000 images across 15 classes with over 87% accuracy.

## 📂 Project Structure
- `app.py`: Streamlit web app for image classification
- `model.py`: Model architecture definition
- `final_model.h5`: Trained model weights
- `evaluate_model.py`: Accuracy and evaluation code
- `confusion_matrix.py`: Confusion matrix visualization
- `requirements.txt`: Required Python packages

## 🧠 Model Details
- Architecture: MobileNetV2
- Classes: 15 animal types (Bear, Bird, Cat, etc.)
- Input Size: 224x224
- Accuracy: ~87%

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run app.py

