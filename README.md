# Odoru - A Real-Time Dance Companion Web App

This repository serves as our group's submission to the HackWashU Hackathon 2024. This web app allows users to dance alongside a dancer video displayed in the web UI, while their movements are captured via a webcam. The app detects the user's skeleton in real-time using a computer vision model, compares their movements to the dancer's, and provides a time-based score based on the similarity of their skeleton moves. The scores are continuously updated throughout the dance session.

## Features

- Video Streaming: Users can select a dance video and view it while recording their dance through a webcam.
- Real-Time Pose Detection: The app uses machine learning and computer vision models to detect the user’s skeleton in real-time from the video stream.
- Motion Similarity Scoring: The app calculates scores by comparing the user's skeletal movements with the reference dance video, providing feedback on their performance.
- Responsive UI: The app provides a seamless interface for users to interact with the video, record themselves, and receive ongoing feedback during their performance.

## Division

Master Division

## Link to Demo Video

https://drive.google.com/file/d/1uV0HPopR7rYMq_374srzIFqs1AVC01Rd/view?usp=sharing

## Team Members

**Weizhi Du**
- Sophomore
- Majors: Computer Science & Financial Engineering
- Backend & Coordinator

[![text](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/duw)

**En "Peter" Wei**
- First-year
- Majors: Computer Science
- Modeling

**Zheyuan Wu**
- Junior
- Majors: Math + Computer Science
- Frontend

[![text](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/zheyuan-wu-742b1a227/)


## Technical Stacks

- Frontend: Next.js
- Backend: Flask, Flask-SocketIO
- Computer Vision: OpenCV, MediaPipe
- Real-time Communication: Socket.IO
- Python Packages: Numpy, Pygame

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/weizhi-du/odoru.git

2. **Enter the repository:**

   ```bash
   cd oduru
   
3. **Set up virtual environment:**

   ```bash
   sudo pip install virtualenv
   virtualenv .venv
   source .venv/bin/activate
   
4. **Install required packages:**

   ```bash
   pip install -r requirements.txt

5. **Open with UI:**

   ```bash
   python app.py
   
This program will be hosted at https://127.0.0.1:5000/


![GitHub last commit](https://img.shields.io/github/last-commit/weizhi-du/odoru)

<!-- ![GitHub license](https://img.shields.io/github/license/weizhi-du/odoru) -->
