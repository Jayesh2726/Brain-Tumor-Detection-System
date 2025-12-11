# 🧠 Brain Tumor Detection Using CNN | Flask Web Application

This project is a **Deep Learning–based Brain Tumor Detection System** that uses **Convolutional Neural Networks (CNN)** to classify MRI brain images as **Tumor** or **No Tumor**.
A simple and clean **Flask web application** allows users to upload MRI images and instantly get prediction results.

---

## 🚀 Features

* 🔍 **CNN model** trained on MRI brain tumor dataset
* 📤 **Image upload interface** (HTML + CSS)
* ⚡ Instant **Tumor / No Tumor prediction**
* 🎨 Clean UI & easy-to-use Flask web interface
* 📁 Supports JPG, PNG, JPEG formats
* 🧪 Pretrained model loaded using TensorFlow/Keras
* 🖥️ Localhost run support (Flask backend)

---

## 🧠 Tech Stack

### **Frontend**

* HTML5
* CSS3

### **Backend**

* Python
* Flask

### **Deep Learning**

* TensorFlow / Keras
* NumPy
* OpenCV
* PIL (Pillow)

---

## 📸 How It Works

1. User uploads an MRI image from the home page
2. The image is resized to **64×64 pixels**
3. Preprocessing is applied
4. The CNN model predicts the class
5. Output is displayed as:

   * **Yes Brain Tumor**
   * **No Brain Tumor**

---

## 📁 Project Structure

```
BrainTumorDetection/
│
├── app.py
├── BrainTumor10Epochs.h5
├── /templates
│    └── index.html
├── /static
│    └── style.css
├── /uploads
└── README.md
```

---

## ▶️ How to Run the Project

### **1. Clone this repository**

```
git clone https://github.com/your-username/Brain-Tumor-Detection-CNN.git
cd Brain-Tumor-Detection-CNN
```

### **2. Install dependencies**

```
pip install -r requirements.txt
```

### **3. Run the Flask app**

```
python app.py
```

### **4. Open in browser**

```
http://127.0.0.1:5000/
```

---

## 📊 Model Information

* Model Type: **Convolutional Neural Network (CNN)**
* Image Size: **64×64**
* Activation: ReLU, Softmax
* Loss Function: Categorical Crossentropy
* Optimizer: Adam
* Epochs: 10

---

## 📷 Sample Output

* ✔️ Uploaded MRI image
* ✔️ Model Prediction
* ✔️ Confidence (optional – add if needed)

---

## 💡 Future Enhancements

* Add confidence/probability score
* Add Grad-CAM heatmap
* Dark/Light theme
* Deploy on Render / Railway / AWS
* Improve accuracy by training more epochs

---

## 👨‍💻 Author

**Jayesh Magare**

* Data Analyst & Machine Learning Enthusiast
* Skilled in Python, CNN, Flask, Data Science

---

