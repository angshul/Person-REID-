# Results

Model: ResNet50
Loss: Softmaxloss

On market1501 Test-set: Rank1:85.74 Rank5: 95.19 Rank10: 96.94 mAP: 67.3

======================================================================================================================

Model: ResNet50
Loss:Softmax Loss
Added: Random Erasing

On market1501 Test-set: Rank1:88.15	Rank5:95.8432	Rank10:97.1793	mAP:72.3722

======================================================================================================================

Model: ResNet50
Loss:Triplet Loss

On Val-set provided by TA: top1: 89.33, top5: 94.67, top10: 97.33, mAP: 74.59
On market1501 Test-set: Rank1:89	Rank5:93.4	Rank10:97	mAP:72.33

======================================================================================================================

Model: DenseNet121
Loss:Triplet Loss (No Random erasing )

Val-Set provided by TA: top1: 86.67, top5: 95.33, top10: 99.33, mAP: 76.28

======================================================================================================================

Model: DenseNet121
Loss: Triplet Loss

On Val-set provided by TA: top1: 92.00, top5: 98.00, top10: 99.33, mAP: 78.68
On market1501 Test-set: top1: 89.8, top5: 96.30, top10: 98.33, mAP: 75.2

======================================================================================================================

Model: MidResNet50 
Loss: Triplet Loss

On Val-set provided by TA: top1: 88.00, top5: 94.567, top10: 94.67, mAP: 73.87

======================================================================================================================

 DenseNet121 gave best result, so we added softmax loss and ran all the experiments on denset121 only. 


| (triplet_loss/softmax_loss) weightage | top1 (val/test) % | top5 (val/test) % | top10 (val/test) % | mAP (val/test) % |
|---------------------------------------|-------------------|-------------------|--------------------|------------------|
| 0.9/0.1                               | 90/90.5           | 95.33/96.7        | 96.67/98.01        | 78.53/77.27      |
| 0.85/0.15                             | 94/89.9           | 97.33/96          | 99.33/97.6         | 79.46/76.82      |
| 0.8/0.2                               | 91.33/90.1        | 98/96.46          | 97.67/97.86        | 78.05/76.878     |
| 0.2/0.8                               | 91.33/91.4        | 97.33/97          | 98.67/98.18        | 79.09/78.8       |
| 0.1/0.9                               | 91.33/92          | 96.67/97.2        | 98/98.2            | 80.11/78.25      |
| 0.1/0.9 - Features Normalize          | **93.33/91.7**       | 97.33/96.88       | 98.67/98           | **81.74/78.97**     |

A .pth file has been uploaded at https://drive.google.com/open?id=1AT-x3HCVWAuZNEqSFvqbpWjnF2mzBYx7 