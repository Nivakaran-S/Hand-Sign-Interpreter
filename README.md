# ✋ Hand Gesture Recognition & Data Collection System  

A computer vision–based system that detects hand gestures in real time, collects training data, and classifies gestures using a deep learning model.  
Built with **OpenCV**, **cvzone**, and **TensorFlow/Keras**, this project demonstrates the full pipeline — from dataset creation to real-time gesture recognition.

---

## 🚀 Features  

### 🔹 **Real-time Hand Tracking**  
- Uses `cvzone.HandTrackingModule` to detect hands and track bounding boxes with high accuracy.  
- Processes only one hand at a time for cleaner dataset creation and classification.  

### 🔹 **Automated Dataset Creation** (`dataCollection.py`)  
- Crops and resizes detected hands into a fixed 300×300 image.  
- Pads the image to maintain aspect ratio.  
- Stores images in class-specific folders for easy model training.  
- Saves files with timestamps for unique naming.  

### 🔹 **Gesture Classification** (`app.py`)  
- Loads a pre-trained Keras model (`model.h5`) with corresponding label mapping (`labels.txt`).  
- Classifies live video input into trained gesture classes.  
- Displays the predicted gesture label directly on the video feed.  

---

## 📂 Project Structure  

```bash 

.
├── app.py # Real-time gesture classification
├── dataCollection.py # Hand image data collection
├── model/
│ ├── model.h5 # Trained gesture classification model
│ ├── labels.txt # Label mapping for gestures
├── images/
│ ├── A/ # Sample class folder for gesture "A"
│ ├── B/ # Sample class folder for gesture "B"
│ └── Z/ # Sample class folder for gesture "Z"
└── requirements.txt # Python dependencies

```


---

## 🛠️ Tech Stack  

- **Programming Language:** Python 3.x  
- **Computer Vision:** OpenCV, cvzone  
- **Machine Learning:** TensorFlow/Keras  
- **Utilities:** NumPy, math, time  

---

## ⚙️ How It Works  

1. **Data Collection Phase**  
   - Run `dataCollection.py` to record gestures.  
   - Press **`s`** to save a cropped & padded image of the detected hand.  
   - Organize images into class folders (e.g., `images/A/`, `images/B/`).  

2. **Model Training Phase** *(not included in repo but easy to implement)*  
   - Train a CNN model on collected gesture images.  
   - Save the trained model as `model/model.h5` and labels as `labels.txt`.  

3. **Real-time Classification Phase**  
   - Run `app.py`.  
   - The camera feed will show the detected hand and predicted gesture label in real time.  

---

## 📦 Installation  

```bash

# Clone the repository
git clone https://github.com/yourusername/hand-gesture-recognition.git
cd hand-gesture-recognition

# Install dependencies
pip install -r requirements.txt

```

--- 

## ▶️ Usage
### Collect Data
```bash 

python dataCollection.py

```

- Adjust folder variable in the script to choose where images are saved.
- Press s to save images for the current gesture.


### Run Gesture Recognition

```bash

python app.py

```

- Make sure model/model.h5 and model/labels.txt are present.
- The system will detect and classify gestures in real time.

## 📸 Example Output

### Real-Time Detection:
- Displays bounding box around detected hand.
- Shows cropped gesture image (ImageCrop) and preprocessed padded image (ImageWhite).
- Prints predicted label above the bounding box.

## 🌟 Future Improvements
- Implement multi-hand support.
- Add more gestures for richer vocabulary.
- Integrate sign language recognition.
- Deploy as a web app using Flask or FastAPI.


## 💡 Why This Project Stands Out
- End-to-End Pipeline: Covers the entire lifecycle from dataset creation to deployment-ready classification.
- Clean & Modular Code: Separate scripts for data collection and classification.
- Real-Time Performance: Optimized for fast processing and minimal lag.
- Portfolio-Ready: Shows computer vision, deep learning, and project structuring skills.
