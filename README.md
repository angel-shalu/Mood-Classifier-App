# 🌈 Mood Classifier App

“Let your face tell your feelings — AI will read them.” 😊



# 🧠 Overview

The Mood Classifier App is a sleek Streamlit-based web app that uses Deep Learning to predict whether a person in an image is Happy 😄 or Not Happy 😐.
You can upload an image or capture one using your webcam — and the app will instantly reveal the mood with confidence levels!


💡 Ideal for emotion-based projects, sentiment analytics, and facial recognition demonstrations.


# ✨ Key Features

✅ AI-Powered Mood Prediction – Detects “Happy” or “Not Happy” from face images

📷 Dual Input Modes – Upload image or use your camera in real-time

📊 Prediction History – Stores your recent 10 predictions with confidence scores

🖼️ Dataset Gallery – Displays example images from your trainning/ dataset

⚙️ Model Management – Upload, reload, and verify model weights dynamically

💻 Modern UI – Minimal, clean, and responsive Streamlit interface


# 🧩 Tech Stack

Technology	Role

🐍 Python 3.10+	Core programming language

⚙️ Streamlit	Front-end web framework

🧠 TensorFlow / Keras	Deep learning backend

🖼️ Pillow (PIL)	Image handling and preprocessing

🗂️ pathlib, os	File & directory management



# 🗂️ Project Structure
mood-classifier-app/
│
├── app.py                  # 🎯 Main Streamlit application

├── model.py                # 🧠 Model class (MoodModel)

├── mood_weights.h5         # ⚙️ Pre-trained model weights

├── trainning/

│   ├── happy/              # 😀 Happy face images

│   └── not happy/          # 😐 Not happy face images

├── requirements.txt        # 📦 Project dependencies

└── README.md               # 📝 Documentation


# ⚙️ Installation & Setup
🔹 Step 1: Clone the Repository
git clone https://github.com/<your-username>/mood-classifier-app.git
cd mood-classifier-app

🔹 Step 2: Create a Virtual Environment
python -m venv venv
venv\Scripts\activate        # on Windows
source venv/bin/activate     # on Mac/Linux

🔹 Step 3: Install Dependencies
pip install -r requirements.txt

🔹 Step 4: Run the App
streamlit run app.py


Then open the local URL displayed in your terminal (usually http://localhost:8501).



# 🧠 How It Works

The MoodModel (from model.py) loads a pre-trained neural network.

You upload or capture an image using your webcam.

The model predicts the mood and displays:

🏷️ Label: “Happy” or “Not Happy”

📈 Confidence Score: Probability value between 0–1

Recent predictions are saved for reference in the sidebar.


# 🖼️ App Pages

Page	Description

🏠 Home	Welcome screen with app overview

📷 Mood Detection	Upload or capture an image and get predictions

🖼️ Dataset Samples	Browse example training images

⚙️ Model Info	Manage weights and view model details


# 📦 requirements.txt
 Here’s a sample requirements.txt to include:

streamlit==1.38.0

Pillow==10.0.0

tensorflow==2.16.1


# 🧾 Sample Dataset Layout

trainning/

├── happy/

│   ├── img1.jpg

│   ├── img2.png

│   └── ...

└── not happy/

    ├── img1.jpg
    
    ├── img2.png
    
    └── ...
    


🟢 You can add any number of images in each folder for testing or retraining.

# 🔍 Model Information

Property	Description

Model Type	Binary Image Classifier

Classes	Happy, Not Happy

Input	Face or portrait image

Output	Predicted class + confidence score

Threshold	0.5 (binary decision boundary)


# 💡 Future Enhancements

✨ Add support for multi-emotion classification (e.g., sad, angry, surprised)

✨ Integrate OpenCV face detection before prediction

✨ Display Grad-CAM heatmaps for explainable AI

✨ Add dark/light theme toggle in the UI

# 🧑‍💻 Developed By

Shalini Kumari
📧 shalinikumari8789@gmail.com

💼 LinkedIn
 | 💻 GitHub

# Developed with ❤️ using Streamlit, TensorFlow, and Python.
