from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision import transforms

from train2 import ImagesDataset, ScaleAndPadToSquare, CropCenterSquare

DATA_PATH = Path('./vk-made-sports-image-classification')
SAVE_PATH = Path("./models")

NUM_CLASSES = 30
MODEL_NAME = "efficientnet_v2_s"
BATCH_SIZE = 64

SQUARE_SIZE = 360  # 224 320 360
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


if __name__ == '__main__':
    train_df = pd.read_csv(DATA_PATH / 'train_fix270.csv')  # check correct labels order!
    test_df = pd.read_csv(DATA_PATH / 'test.csv')
    labels = train_df.label.value_counts().index.tolist()
    label2number = {k: v for v, k in enumerate(labels)}
    number2label = {v: k for k, v in label2number.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = models.densenet169(weights='IMAGENET1K_V1')  # !!!
    # model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES, bias=True)
    model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(model.classifier[1].in_features, NUM_CLASSES, bias=True)
    )

    # densenet161
    # _74.65_80.26.pth
    # with open(SAVE_PATH / f"{MODEL_NAME}_0.49_0.43.pth", "rb") as fp:  # models_0
    with open(SAVE_PATH / f"{MODEL_NAME}_ep7_t0.21_v0.22.pth", "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)
    model.to(device)

    test_transforms = transforms.Compose([
        ScaleAndPadToSquare(SQUARE_SIZE),
        # CropCenterSquare(SQUARE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    test_dataset = ImagesDataset(
        data_dir=DATA_PATH / 'test',
        data_df=test_df,
        split='test',
        transform=test_transforms,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        num_workers=4, pin_memory=True,
        shuffle=False, drop_last=False
    )

    # # train check (find wrong labels)
    # train_dataset = ImagesDataset(
    #     data_dir=DATA_PATH / 'train',
    #     data_df=train_df,
    #     split='test',  # no 'val' column
    #     transform=test_transforms,
    # )
    # train_loader = DataLoader(
    #     train_dataset, batch_size=BATCH_SIZE,
    #     num_workers=4, pin_memory=True,
    #     shuffle=False, drop_last=False
    # )


    def predict(model, loader, device):
        model.eval()
        predictions = np.zeros((len(loader.dataset), NUM_CLASSES))
        for i, batch in enumerate(tqdm(loader, total=len(loader), desc="test prediction...", position=0, leave=True)):
            images = batch[0].to(device)

            with torch.no_grad():
                pred_logits = model(images).cpu()  # B x NUM_CLASSES

            predictions[i * loader.batch_size: (i + 1) * loader.batch_size] = pred_logits.numpy()

        return predictions


    test_predictions = predict(model, test_loader, device)
    # test_predictions = predict(model, train_loader, device)  # train check

    test_labels = [number2label[np.argmax(x)] for x in test_predictions]
    submission_df = pd.DataFrame({'image_id': test_df.image_id, 'label': test_labels})

    # # train check
    # # submission_df = pd.DataFrame({'image_id': train_df.image_id, 'label': test_labels})
    # submission_df = train_df.copy()
    # submission_df['pred_label'] = test_labels

    submission_df.to_csv('submission.csv', header=True, index=False)

    test_probs = softmax(test_predictions, axis=1)
    print(test_probs.shape)
    with open('test_probs.npy', 'wb') as f:
        np.save(f, test_probs)
    # with open('test_probs.npy', 'rb') as f:
    #     test_probs = np.load(f)
