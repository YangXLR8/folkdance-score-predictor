<h1 align="center">TIKLOS: Folkdance Score Predictor Model</h1>
<h3 align="center">Computer Vision Final Project</h3>

<p align="center">
  <img width="700" src="" alt="cli output"/>
</p>

## Description

This final project calculates the difference between key reference landmarks in an instructor's video and the corresponding landmarks in a student's video to determine the predicted score, using MediaPipe's Pose Landmarker Heavy model.


## Features

- **Draw Landmarks on Reference Video**: Visualizes landmarks on detected persons and the expected connections between them.
- **Save Landmarks and Timestamps**: Records landmarks and their corresponding timestamps.
- **Draw Landmarks on Student Video**: Visualizes landmarks on the student's video.
- **Compute Difference**: Calculates the difference between landmarks in the reference and student videos.
- **Display Score**: Displays the computed score based on the student's video.

## Project Structure

- `false-vids/`: Contains the non-tiklos(false) videos
- `images/`: images folder.
- `prof/`: Contains instructor(reference) videos
- `student/`: Contains student(to be scored) videos
- `mediapipe.ipynb/`: Main notebook that contains the project 
- `Tiklos Folkdance Predictor Score.pdf`: summary presentation

## Requirements

- Python 3
- OpenCV (`cv2`)
- MediaPipe
- Pose Landmarker Heavy Model (`pose_landmarker.task`)
- Instructor dance Video
- Student dance video


## Usage

1. Run the mediapipe.ipynb 

## Results

This project displays the results at the `mediapipe.ipynb`.

## Colloborators

👤 **Camarista, Charisse** - Github: [@cha2501](https://github.com/cha2501)


👤 **Sardañas, Reah Mae** - Github: [@YangXLR8](https://github.com/YangXLR8)

👤 **Valente, Nhilbert Jay** - Github: [@NhilbertJayValente](https://github.com/NhilbertJayValente)