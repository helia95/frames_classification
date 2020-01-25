# Trimmed action recognition and temporal action segmentation in full-length videos.

The dataset can be downloaded from this [this link](https://drive.google.com/uc?export=download&id=1ncmqWLctmvecIXBdVng5cvbROoTWFSpE).


## Trimmed action recognition
Implementation of RNNs for classification of a sequence of frames with a unique action (i.e. classification many-to-one).

>> The achieved test accuracy 0.47

## Temporal action segmentation
Given a full-length video, segment it recognising the different actions (i.e. classification many-to-many).

>> The achieved test accuracy 0.56


Results for *continental breakfast* video, the first row represent the prediction, the second the ground truth.


<p align="center">
<img                
src=prediciton.png>
</p>
