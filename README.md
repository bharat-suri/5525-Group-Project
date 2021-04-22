# 5525-Group-Project

## RUN

```
python train_finegrained_multilabel.py --device YOUR_GPU_ID --batch-size 8 --lr 3e-5 --epochs 10
```

## EVALUATE

You do need to modify the save checkpoint list in line 219. The last outout line returns the offitial evaluation results.

```
python train_finegrained_multilabel.py 
```
