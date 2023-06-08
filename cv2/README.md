# [VK MADE] Optical Character Recognition

The task is to recognize text in images.

## Dataset Description

The Optical Character Recognition dataset consists of approximately 330,000 images. The dataset is split into training data and test data, and the test data is further split in halves for public and private leaderboard. Private leaderboard will be available right after the contest deadline.

### Dataset File Structure

The dataset is structured in the following format:

```
dataset/
|-- train_labels.csv
|-- sample_submission.csv
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

The target metric is LevenshteinMean.
