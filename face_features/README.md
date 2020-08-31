# Face Feature Extraction

This is a lightweight facial feature extractor extracting a 164 dimensional facial feature vector from an image. It includes the corners of the mouth and eyes, tip of the nose, pupil movement, coordinates of all corners from the eyes to calculate the center, origin, area, height, width, and a ratio that can indicate whether an eye is closed or not. Based on the relative spatial estimation that the iris occupies in relation to the ocular surface is used to calibrate the distance between camera and subject. In order to determine the direction a person is looking at, a vertical and horizontal ratio between 0 and 1 is calculated, so that a value of 0.0 reflects the top, 0.5 the middle, and 1.0 the bottom level. A binary value that represents left versus right oriented gaze is also derived by comparing this value with fixed thresholds. The horizontal threshold is set to $\beta = .35$ ($<=$ right, $>= 1-\beta$ left).

## Acknowledgement
This feature extraction is heavily based on OpenCV, Dlib and the webcam-based eye tracking system of https://github.com/antoinelame/GazeTracking. All credits for this great implementation! 


## Installation

1. Install dependencies:

```
pip install -r requirements.txt
```
Further Dlib requires Boost, Boost.Python, CMake and X11/XQuartx [help](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/).

2. Place the pretrained models in the folder gaze_tracking/trained_models
3. Specify the DGW data folder


## Run
### Extraction
```
 usage: run_extraction_for_dgw.py [-h] [-d DATA_PATH] [-type EXTRACTION_TYPE]
 
 optional arguments:
   -h, --help            show this help message and exit
   -d DATA_PATH, --data_path DATA_PATH
                         specify which data path
   -type EXTRACTION_TYPE, --extraction_type EXTRACTION_TYPE
                         specify if mixed (hog + cnn) or purely cnn extraction
```

### SVMs
run_svm_on_face_features.py

## Licensing
Several licenses of the individual code parts are valid! Please check the mentioned frameworks and the license file in the subfolder gaze_tracking for further information.