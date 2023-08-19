from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from crnn import CRNN
from data_analysis import get_symbols_hist
from dataset import RecognitionDataset, Cv2Resize, collate_fn_rotate
from train_utils import bruteforce_rotation_model


EXPERIMENT_NAME = 'crnn_effnet'
IMG_WIDTH = 320
IMG_HEIGHT = 64
BATCH_SIZE = 64
NUM_WORKERS = 4

torch.manual_seed(0)


if __name__ == '__main__':
    data_path = Path('./ocr_contest_data')
    test_data_path = Path(data_path / 'test')

    labels_df = pd.read_csv(data_path / 'extended_train_labels_st5_ft3.csv')
    alphabet = ''.join([ch for ch, _ in get_symbols_hist(labels_df)])
    test_df = pd.read_csv(data_path / f'test_data.csv')

    transformations_val_test = transforms.Compose([
        Cv2Resize(output_size=(IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])

    train_rotation_dataset = RecognitionDataset(
        files_dir=labeled_data_path, data_df=labels_df,
        alphabet=alphabet, transforms=transformations_val_test,
        split='train', train_size=1.0, rotate_4=True  # rotate !!!
    )

    test_rotation_dataset = RecognitionDataset(
        files_dir=test_data_path, data_df=test_df,
        alphabet=alphabet, transforms=transformations_val_test,
        split='test', rotate_4=True  # rotate !!!
    )

    train_rotation_dataloader = DataLoader(
        train_rotation_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        drop_last=False, collate_fn=collate_fn_rotate
    )

    test_rotation_dataloader = DataLoader(
        test_rotation_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        drop_last=False, collate_fn=collate_fn_rotate
    )

    # print(next(iter(train_check_dataloader))['image'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    crnn = CRNN(alphabet=alphabet, cnn_input_size=(IMG_HEIGHT, IMG_WIDTH), cnn_output_len=30,
                rnn_hidden_size=128, rnn_num_layers=2, rnn_dropout=0.3, rnn_bidirectional=True)
    crnn.to(device)

    rotation_df = bruteforce_rotation_model(crnn, EXPERIMENT_NAME, device, train_rotation_dataloader, labeled_data=True)
    rotation_df.to_csv('labeled_data_rotation.csv', header=True, index=False)
    
    rotation_max_df = rotation_df.loc[rotation_df.groupby('img', sort=False).logit.idxmax()].reset_index(drop=True)
    rotation_max_df.to_csv('labeled_data_rotation_max.csv', header=True, index=False)
