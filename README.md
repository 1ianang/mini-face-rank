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
## 7. Result
```
samples\a1.jpeg: 4.0690
samples\a2.jpeg: 4.1459
samples\a3.jpeg: 3.7766
samples\a4.jpeg: 4.4120
samples\a5.jpg: 4.9525
samples\a6.jpg: 4.5757
samples\a7.jpg: 3.9572
samples\b1.jpg: 2.0240
samples\b2.jpg: 2.1274
samples\b3.jpg: 0.7316
samples\b4.jpg: 2.2222
```
A sample of these images:
![](samples/a1.jpeg!width=100)
![](samples/a2.jpeg!width=100)
![](samples/b1.jpeg!width=100)
![](samples/b2.jpeg!width=100)
## 8. More face ranking projects
- https://github.com/fendouai/FaceRank