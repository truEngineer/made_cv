# [VK MADE] Sports Image Classification

The task is to classify sport images into 30 different sport classes. 
The dataset may contain markup errors and it is expected from the students to come up with solutions for these issues.

## Dataset Description

The Sports Image Classification Contest dataset consists of approximately 65,000 images divided into 30 different sport classes.
The classes are imbalanced, with varying numbers of images per class.
The dataset is split into training data (70%) and test data (30%), and the test data is further split in halves for public and private leaderboard.

### Dataset File Structure
The dataset is structured in the following format:

```
dataset/
|-- train.csv
|-- test.csv
|-- train/
|   |-- {image_1}.jpeg
|   |-- {image_2}.jpeg
|   |-- ...
|-- test/
|   |-- {image_1}.jpeg
|   |-- {image_2}.jpeg
|   |-- ...
```

## Evaluation Process

The evaluation of the models will be based on the micro F1-score, with the participant with the highest F1-score declared the winner.
The test data split between public and private and the split is only known to the organizers and will not be revealed to the participants.
