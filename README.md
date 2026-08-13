# 😷 Face Mask Detection

AI-powered face mask detection system using **OpenCV** and a **CNN deep learning model**, featuring a **Streamlit web interface** for image-based and real-time detection.

## 📌 Project Overview

This project detects whether a person is wearing a face mask using computer vision and deep learning.

The system uses:

* **OpenCV** for face detection
* **Convolutional Neural Network (CNN)** for mask classification
* **Streamlit** for the interactive web interface
* **Keras/TensorFlow** for loading and using the trained deep learning model

The application supports both **image detection** and **real-time webcam detection**.

## ✨ Features

* 😷 Detects whether a face is wearing a mask
* 📷 Image-based mask detection
* 🎥 Real-time webcam detection
* 🧑 Face detection using OpenCV Haar Cascade
* 🧠 CNN-based mask classification
* 🌐 Interactive Streamlit interface
* ⚡ Simple and easy-to-use application

## 🛠️ Technologies Used

| Technology         | Purpose                               |
| ------------------ | ------------------------------------- |
| Python             | Core programming language             |
| OpenCV             | Face detection and computer vision    |
| TensorFlow / Keras | Deep learning model                   |
| NumPy              | Numerical operations                  |
| Streamlit          | Web application interface             |

## 🧠 Machine Learning Model

The project uses a **Convolutional Neural Network (CNN)** trained to classify detected faces into mask and non-mask categories.

The CNN architecture includes:

* Convolutional layers
* Max Pooling layers
* Flatten layer
* Fully Connected (Dense) layers
* Output classification layer

The model is trained using image data with preprocessing and data augmentation techniques such as:

* Rescaling
* Shearing
* Zooming
* Horizontal flipping

The trained model is saved as:

```text
Mask.h5
```

## 👁️ Face Detection

Before classification, the application detects faces using an OpenCV Haar Cascade classifier.

The cascade file used by the project is:

```text
face.xml
```

The detected face regions are then passed to the CNN model for mask classification.

## 📂 Project Structure

```text
Face_Mask_Detection/
│
├── app.py
├── backend.py
├── face.xml
├── requirements.txt
├── README.md
├── Mask.h5
```
> The exact project structure may vary depending on the final files uploaded to the repository.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/vansh0402/Face_Mask_Detection.git
```

### 2. Navigate to the project directory

```bash
cd Face_Mask_Detection
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```bash
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔄 How It Works

```text
Input Image / Webcam
        ↓
   Face Detection
        ↓
  Face Region Extracted
        ↓
    Image Preprocessing
        ↓
       CNN Model
        ↓
 Mask / No Mask Prediction
        ↓
     Display Result
```

## 📷 Detection Modes

### Image Detection

Upload an image containing a person. The application detects the face and predicts whether the person is wearing a mask.

### Real-Time Detection

The application can use a webcam to continuously detect faces and classify them as:

* **Mask**
* **No Mask**

## 📦 Requirements

The main Python libraries used in this project include:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
opencv-python
tensorflow
keras
streamlit
```

For the exact dependencies required to run the application, refer to:

```text
requirements.txt
```

## 🚀 Future Improvements

* Improve model accuracy with a larger and more diverse dataset
* Add confidence scores to predictions
* Improve real-time detection performance
* Deploy the application online
* Add support for multiple faces simultaneously
* Experiment with more advanced CNN architectures
* Add model performance metrics and evaluation visualizations

## 👨‍💻 Author

**Vansh**

Data Science / Data Analytics Fresher

GitHub: `https://github.com/vansh0402`

---

⭐ If you found this project useful, consider giving the repository a star!
