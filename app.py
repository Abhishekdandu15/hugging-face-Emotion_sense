import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import cv2
from deepface import DeepFace
import tempfile
import os
import cv2

# User authentication storage (simple dictionary for now)
users = {"admin": "password123"}

# Emoji mapping for emotions
EMOJI_MAP = {
    "angry": "😡",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😃",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

# Function to register a new user
def register(username, password):
    if username in users:
        return "❌ Username already exists. Please try another."
    users[username] = password
    return "✅ Registration successful! You can now log in."

# Function to login
def login(username, password):
    if username in users and users[username] == password:
        return "✅ Login successful! You can now proceed."
    return "❌ Invalid credentials. Please try again."

# Function to train model and display dataset insights
def train_model(file):
    df = pd.read_csv(file.name)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    class_report = classification_report(y_test, predictions)
    joblib.dump(model, "emotion_model.pkl")
    
    # Display only top 4 rows of dataset
    top_4_data = df.head(4).to_string()
    
    # Plot dataset overview
    plt.figure(figsize=(8, 5))
    df[y.name].value_counts().plot(kind='line', marker='o', color='skyblue')
    plt.xlabel("Emotions")
    plt.ylabel("Score")
    plt.title("📊 Dataset Emotion Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("dataset_plot.png")
    
    return f"<h2 style='font-size:24px;'>✅ Model trained successfully with accuracy: {accuracy:.2f} 🎯</h2>", "dataset_plot.png", f"📊 Top 4 Rows:\n{top_4_data}", f"📑 Classification Report:\n{class_report}"

# Function to predict emotion from video and generate line graph
def predict_emotion(video):
    model = joblib.load("emotion_model.pkl")
    
    temp_video_path = video.name
    cap = cv2.VideoCapture(temp_video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    if duration < 2 or duration > 20:
        return "⚠️ Error: Video duration must be between 2 and 20 seconds. ⏳", None, None, None
    
    frame_results = {}
    
    frame_interval = int(fps)
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_id % frame_interval == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            analysis = DeepFace.analyze(img_rgb, actions=["emotion"], enforce_detection=False)
            if analysis:
                for key, value in analysis[0]['emotion'].items():
                    frame_results[key] = frame_results.get(key, []) + [value]
        
        frame_id += 1
    
    cap.release()
    
    avg_scores = {key: np.mean(values) for key, values in frame_results.items()}
    
    predicted_emotion = max(avg_scores, key=avg_scores.get)
    predicted_emoji = EMOJI_MAP.get(predicted_emotion, "❓")
    
    # Plot emotions as bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(list(avg_scores.keys()), list(avg_scores.values()), color='skyblue')
    plt.xlabel("Emotions")
    plt.ylabel("Score")
    plt.title("📊 Emotion Analysis Over Video")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("emotion_plot.png")
    
    # Generate line graph for emotions
    plt.figure(figsize=(10, 5))
    plt.plot(list(avg_scores.keys()), list(avg_scores.values()), marker='o', linestyle='-')
    plt.xlabel("Emotions")
    plt.ylabel("Score")
    plt.title("📈 Emotion Trends Over Video")
    plt.grid()
    plt.savefig("video_trend_plot.png")
    
    return f"{predicted_emotion} {predicted_emoji}", "emotion_plot.png", "video_trend_plot.png"

# Create Gradio Interfaces
register_interface = gr.Interface(
    fn=register,
    inputs=[gr.Textbox(label="Username"), gr.Textbox(label="Password", type="password")],
    outputs=gr.Textbox(label="Registration Status"),
    title="🔑 User Registration"
)

login_interface = gr.Interface(
    fn=login,
    inputs=[gr.Textbox(label="Username"), gr.Textbox(label="Password", type="password")],
    outputs=gr.Textbox(label="Login Status"),
    title="🔐 User Login"
)

train_interface = gr.Interface(
    fn=train_model,
    inputs=gr.File(label="📂 Upload Dataset (CSV)"),
    outputs=[gr.HTML(label="📊 Training Status"), gr.Image(label="📈 Dataset Overview"), gr.Textbox(label="📊 Top 4 Frames"), gr.Textbox(label="📑 Classification Report")],
    title="🎓 Train Emotion Model"
)

predict_interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.File(label="🎥 Upload Video for Emotion Detection"),
    outputs=[gr.Label(label="🎭 Predicted Emotion"), gr.Image(label="📈 Emotion Score Graph"), gr.Image(label="📈 Video Emotion Trend")],
    title="🎭 Emotion Detection from Video"
)

# Launch Gradio Tabs
gr.TabbedInterface([register_interface, login_interface, train_interface, predict_interface], ["📝 Register", "🔑 Login", "🛠 Train Model", "🎭 Predict Emotion"]).launch()
