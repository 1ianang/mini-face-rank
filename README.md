## Just a simple face attractiveness ranker

### 1. Dependencies
- keras2.1.2
- face-recognition1.0.0

## 2. Dataset
SCUT-FBP5500 is a diverse benchmark database for multi-paradigm facial beauty prediction, which is collected by SCUT. You can download it from their github: https://github.com/HCIILAB/SCUT-FBP5500-Database-Release.

## 3. Preparing
In this step, a face encoding is extracted from each image and is associated with a label. I use pickle to dump it as `training.pkl`

## 4. Training
```
python3 train.py
```
Model would be saved as `face_rank_model.h5`

## 5. Evaluate
```
python3 evaluate.py
```
## 6. Predict
```
python3 predict.py
```
## 8. More face ranking projects
- https://github.com/fendouai/FaceRank
