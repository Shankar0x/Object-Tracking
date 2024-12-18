# Object Tracking with Kalman Filtering

This repository demonstrates the implementation of object tracking techniques using Kalman filtering. The project includes core functionalities for predicting and updating object positions, evaluating tracking performance, and visualizing results.

## Features

- **Kalman Filtering and DeepSORT for Object Tracking**
  - Implements prediction and update steps to estimate object states in video frames for Kalman Filtering.
  - Inferencing from YOLOv8 object detection module and DeepSORT algorithm for object tracking 
  - Tracks and visualizes object positions with annotated ground truth data.

- **Error Analysis**
  - Computes signed and unsigned Euclidean distances between predicted and ground truth positions.
  - Provides graphical plots to evaluate tracking performance.

- **Visualization**
  - Generates output videos with annotated predicted and ground truth positions.
  - Plots center errors for qualitative and quantitative analysis.
  - Displays the final sum of unsigned distances as an error measure.

## Installation

*Highly recommended method*
Open the Google Colab notebook using this [link](https://drive.google.com/file/d/1wcA4RlwdJ_PLWxzyQPI3KWKeDD0YF5zK/view?usp=sharing)

OR

1. Clone the repository:
   ```bash
   git clone https://github.com/Shankar0x/Object-Tracking
   cd Object-Tracking-main
   ```
2. Open the Object_tracking.ipynb notebook and select a python kernel 

## Demo 
1. Kalman Filtering Object Tracking


![Kalman Filter](assets/Kalman%20Car%204.gif)

2. DeepSORT Object Tracking


![DeepSORT](assets/DeepSort%20Car%204.gif)