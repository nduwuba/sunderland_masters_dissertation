import os
import sys
import numpy as np
import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QLabel,
    QFileDialog, QHBoxLayout, QDesktopWidget, QSizePolicy, QProgressBar,
    QTextEdit
)

from PyQt5.QtCore import Qt, QUrl
from tensorflow.keras.models import load_model
import librosa
from deepface import DeepFace
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Emotion dictionary for both audio and video analysis
emotions = {
    "angry": "😠",
    "disgust": "😖",
    "fear": "😱",
    "happy": "😊",
    "sad": "😢",
    "surprise": "😮",
    "neutral": "😐",
    "calm": "😌"
}


# Function to extract MFCC features from an audio file
def extract_mfcc(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}", e)
        return None
    return mfccs

class SentimentAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.audio_model = load_model('audio_model')  # Ensure the correct path for audio model
        self.initUI()
        self.file_path = None
        self.analysis_requested = False
        self.text_analyzer = SentimentIntensityAnalyzer()
        self.progress_timer = QTimer(self)  # Create a timer for resetting progress bar
        self.progress_timer.timeout.connect(self.resetProgressBar)  # Connect the timer to resetProgressBar method

    def initUI(self):
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        layout = QVBoxLayout(centralWidget)

        headerLabel = QLabel("Sentiment Analyser")
        headerLabel.setAlignment(Qt.AlignCenter)
        headerLabel.setStyleSheet("font-size: 36px; font-weight: bold; margin: 20px;")
        layout.addWidget(headerLabel)

        buttonLayout = QHBoxLayout()

        self.loadButton = QPushButton("Load Media File")
        self.loadButton.clicked.connect(self.loadFile)
        self.loadButton.setStyleSheet("font-size: 18px; font-weight: 500")
        self.loadButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.analyseButton = QPushButton("Analyse")
        self.analyseButton.clicked.connect(self.predictEmotion)
        self.analyseButton.setStyleSheet("font-size: 18px; font-weight: 500")
        self.analyseButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.resetButton = QPushButton("Reset")
        self.resetButton.clicked.connect(self.resetTool)
        self.resetButton.setStyleSheet("font-size: 18px; font-weight: 500")
        self.resetButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        buttonLayout.addWidget(self.loadButton)
        buttonLayout.addWidget(self.analyseButton)
        buttonLayout.addWidget(self.resetButton)

        layout.addLayout(buttonLayout)

        self.progressLabel = QLabel("")
        self.progressLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progressLabel)

        self.progressBar = QProgressBar()
        self.progressBar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progressBar)
        self.progressBar.setStyleSheet("QProgressBar { border: none; border-radius: 5px; color: 'white'; background-color: #f0f0f0; height: 20px; text-align: center; }"
                               "QProgressBar::chunk { background-color: #127af8; color: #127af8; margin: 1px; }")



        self.textEdit = QTextEdit()
        self.textEdit.setPlaceholderText("Input Text Here")  # Add placeholder text
        self.textEdit.setStyleSheet("border: 2px solid #ADD8E6; padding: 5px;")
        layout.addWidget(self.textEdit)

        self.predictionLabel = QLabel("Select a file or input text to predict sentiment")
        self.predictionLabel.setAlignment(Qt.AlignCenter)
        self.predictionLabel.setStyleSheet("font-size: 24px; margin-top: 20px;")
        layout.addWidget(self.predictionLabel)

        self.adjustSizeAndPosition()


    def adjustSizeAndPosition(self):
        self.resize(800, 600)
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # Set size policy to fixed

    def loadFile(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Media Files (*.wav *.mp4 *.avi *.mov)")
        if self.file_path:
            self.predictionLabel.setText("File Loaded Successfully")
            self.textEdit.clear()  # Clear the text box when a file is loaded

    def analyzeAudio(self):
        if self.file_path:
            mfccs = extract_mfcc(self.file_path)
            if mfccs is not None:
                mfccs = np.expand_dims(np.expand_dims(mfccs, 0), -1)
                prediction = self.audio_model.predict(mfccs)
                predicted_class = np.argmax(prediction) + 1  # 1-based index

                # Check if predicted_class is within the valid range
                if predicted_class > 0 and predicted_class <= len(emotions):
                    predicted_emotion = list(emotions.keys())[predicted_class - 1]  # Convert to 0-based index
                    self.predictionLabel.setText(f'Result: {predicted_emotion} {emotions[predicted_emotion]}')
                else:
                    self.predictionLabel.setText('Result: Model output out of expected range')
            else:
                self.predictionLabel.setText('Result: Error processing audio file')

    def analyzeText(self):
        text = self.textEdit.toPlainText()
        if text:
            # Analyze sentiment using NLTK Vader
            sentiment_score = self.text_analyzer.polarity_scores(text)
            if sentiment_score['compound'] >= 0.05:
                sentiment_label = 'positive 😊'
            elif sentiment_score['compound'] <= -0.05:
                sentiment_label = 'negative 😠'
            else:
                sentiment_label = 'neutral 😐'
            self.predictionLabel.setText(f'Result: {sentiment_label}')


    def analyzeVideo(self):
        if self.file_path:
            self.capture = cv2.VideoCapture(self.file_path)
            if not self.capture.isOpened():
                self.predictionLabel.setText("Could not open video file")
                return

            emotions_detected = []

            total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0

            while True:
                ret, frame = self.capture.read()
                if not ret:
                    break

                current_frame += 1
                progress = int((current_frame / total_frames) * 100)
                self.progressBar.setValue(progress)

                temp_image_path = "temp_frame.jpg"
                cv2.imwrite(temp_image_path, frame)  # Save frame as JPEG

                try:
                    analysis = DeepFace.analyze(img_path=temp_image_path, actions=['emotion'], enforce_detection=False)
                    # Adjust handling here to account for analysis being a list
                    if isinstance(analysis, list) and len(analysis) > 0:
                        analysis_first_result = analysis[0]  # Take the first result if there are multiple faces
                        if 'dominant_emotion' in analysis_first_result:
                            dominant_emotion = analysis_first_result['dominant_emotion']
                            emotions_detected.append(dominant_emotion)
                        else:
                            print("Dominant emotion not found in first result of analysis.")
                    else:
                        print("Unexpected analysis result structure:", analysis)
                except Exception as e:
                    print(f"Error during emotion analysis: {e}")

            self.capture.release()
            os.remove(temp_image_path)  # Removing the temporary file after analysis

            if emotions_detected:
                most_frequent_emotion = max(set(emotions_detected), key=emotions_detected.count)
                self.predictionLabel.setText(f'Result: {most_frequent_emotion} {emotions.get(most_frequent_emotion, "🤷")}')   
            else:
                self.predictionLabel.setText("No emotions detected in the video.")
            self.progressBar.setValue(100)
            self.analysis_requested = False

    def predictEmotion(self):
        if self.textEdit.toPlainText():
            self.analysis_requested = False  # Reset the flag
            self.progressBar.setValue(0)  # Reset progress bar
            self.progressLabel.setText("Analysis in Progress...")
            self.predictionLabel.setText("")  # Clear the prediction label
            self.analyzeText()  # Call analyzeText method for text input
            # Start the timer to reset progress bar if prediction lasts more than 1 second
            self.progress_timer.start(1000)  # Start the timer
        elif self.file_path:
            self.analysis_requested = False  # Reset the flag
            self.progressBar.setValue(0)  # Reset progress bar
            self.progressLabel.setText("Analysis in Progress...")
            self.predictionLabel.setText("")  # Clear the prediction label
            if self.file_path.endswith('.wav'):
                self.analyzeAudio()
            elif self.file_path.endswith(('.mp4', '.avi', '.mov')):
                self.analyzeVideo()
            else:
                self.analyzeText()  # Call analyzeText method for text input
            # Start the timer to reset progress bar if prediction lasts more than 1 second
            self.progress_timer.start(1000)  # Start the timer
        else:
            self.predictionLabel.setText("Please load a file or input text first.")

    def resetProgressBar(self):
        self.progressBar.setValue(0)
        self.progressLabel.setText("")
        self.progress_timer.stop()  # Stop the timer after resetting the progress bar

    def resetTool(self):
        self.file_path = None
        self.textEdit.clear()  # Clear the text box
        self.predictionLabel.setText("Please Enter Text or Upload Media")
        self.progressLabel.setText("")
        self.progressBar.setValue(0)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SentimentAnalyzer()
    ex.show()
    sys.exit(app.exec_())
