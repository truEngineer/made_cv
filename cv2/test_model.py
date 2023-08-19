from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from crnn import CRNN
from data_analysis import get_symbols_hist
from dataset import RecognitionDataset, Cv2Resize, collate_fn_recognition
from train_utils import test_model


EXPERIMENT_NAME = 'crnn_effnet'
IMG_WIDTH = 320
IMG_HEIGHT = 64
BATCH_SIZE = 128
NUM_WORKERS = 4

torch.manual_seed(0)


if __name__ == '__main__':
    data_path = Path('./ocr_contest_data')
    test_data_path = Path(data_path / 'test')
  
    # labels_df = pd.read_csv(data_path / 'extended_train_labels_st5_ft3.csv')
    labels_df = pd.read_csv('labeled_data_rotation_max.csv')  # df with image angles info
    alphabet = ''.join([ch for ch, _ in get_symbols_hist(labels_df)])
    # test_df = pd.read_csv(data_path / f'test_data.csv')
    test_df = pd.read_csv('test_data_rotation_max.csv')

    transformations_val_test = transforms.Compose([
        Cv2Resize(output_size=(IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = RecognitionDataset(
        files_dir=test_data_path, data_df=test_df,
        alphabet=alphabet, transforms=transformations_val_test,
        split='test'
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        drop_last=False, collate_fn=collate_fn_recognition
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    crnn = CRNN(alphabet=alphabet, cnn_input_size=(IMG_HEIGHT, IMG_WIDTH), cnn_output_len=30,
                rnn_hidden_size=128, rnn_num_layers=2, rnn_dropout=0.3, rnn_bidirectional=True)
    crnn.to(device)
    test_preds = test_model(crnn, EXPERIMENT_NAME, device, test_dataloader)

    submission_df = pd.DataFrame({'Id': test_df['img'], 'Predicted': test_preds})
    submission_df.to_csv('submission.csv', header=True, index=False)
