# Coconut Ripeness Detection System

This project presents a non-destructive coconut ripeness classification system using both acoustic and image-based sensing. The system integrates audio analysis, image processing, and machine learning to improve classification reliability.

## Project Overview

Coconut ripeness is traditionally identified by manual tapping and visual inspection. This method depends on experience and is not always consistent.

In this project, I developed a fusion-based approach that combines:

- Tapping sound analysis (internal structure)
- Surface image analysis (external features)
- Machine learning classification

By combining both modalities, the system achieves improved robustness compared to single-sensor methods.

## Methodology

1. Record tapping sound using microphone.
2. Capture coconut image using ESP32-CAM.
3. Extract audio features:
   - MFCC
   - Spectral centroid
   - Zero crossing rate
4. Extract visual features:
   - Color-based features
   - Texture-based features
5. Perform feature-level fusion.
6. Train Random Forest classifier.

## Results

- Audio-only model: Strong performance
- Image-only model: Moderate performance
- Fusion model: Improved stability and overall accuracy

## Technologies Used

- Python
- OpenCV
- NumPy
- Librosa
- Scikit-learn
- ESP32-CAM

## Author

Basavaraj Rajendra Patil  
M.Tech Biotechnology, RV College of Engineering
