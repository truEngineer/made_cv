import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms

from crnn import CRNN
from data_analysis import get_symbols_hist
from dataset import RecognitionDataset, Cv2Resize, collate_fn_recognition
from train_utils import count_parameters, train_model

EXPERIMENT_NAME = 'crnn_effnet'
IMG_WIDTH = 320
IMG_HEIGHT = 64

NUM_EPOCHS = 20
BATCH_SIZE = 128
NUM_WORKERS = 4
LEARNING_RATE = 1e-4

np.random.seed(0)
torch.manual_seed(0)


if __name__ == '__main__':
    labels_df = pd.read_csv(data_path / 'extended_train_labels_st5_ft3.csv')
    # labels_df = pd.read_csv('labeled_data_rotation_max.csv')  # df with image angles info

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'{EXPERIMENT_NAME}, current device: {device}')
    alphabet = ''.join([ch for ch, _ in get_symbols_hist(labels_df)])
    print(f'alphabet: {len(alphabet)} symbols')
    crnn = CRNN(alphabet=alphabet, cnn_input_size=(IMG_HEIGHT, IMG_WIDTH), cnn_output_len=30,
                rnn_hidden_size=128, rnn_num_layers=2, rnn_dropout=0.3, rnn_bidirectional=True)
    # freeze layers
    # for child in crnn.features_extractor.cnn.children():
    #     for param in child.parameters():
    #         param.requires_grad = False
    crnn.to(device)

    optimizer = torch.optim.AdamW(crnn.parameters(), lr=LEARNING_RATE, amsgrad=True, weight_decay=1e-2)
    # optimizer = torch.optim.Adam(crnn.parameters(), lr=1e-4, amsgrad=True, weight_decay=1e-5)
    # criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=1 / np.sqrt(10), patience=2, verbose=True, threshold=1e-3
    )

    print(f'{EXPERIMENT_NAME}: {count_parameters(crnn)} parameters')

    transformations_train = transforms.Compose([
        Cv2Resize(output_size=(IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToPILImage(),
        transforms.RandomGrayscale(p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.RandomApply(transforms=[transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.2, 1.2))], p=0.3),
        transforms.RandomRotation(degrees=(-3, 3)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = RecognitionDataset(
        files_dir=labeled_data_path, data_df=labels_df,
        alphabet=alphabet, transforms=transformations_train,
        split='train', train_size=0.8
    )

    transformations_val_test = transforms.Compose([
        Cv2Resize(output_size=(IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToPILImage(),
        # transforms.RandomGrayscale(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])

    val_dataset = RecognitionDataset(
        files_dir=labeled_data_path, data_df=labels_df,
        alphabet=alphabet, transforms=transformations_val_test,
        split='val'
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        drop_last=True, collate_fn=collate_fn_recognition
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        drop_last=False, collate_fn=collate_fn_recognition
    )

    train_history, val_history = train_model(
        crnn, EXPERIMENT_NAME, NUM_EPOCHS, device, optimizer,
        lr_scheduler, train_dataloader, val_dataloader
    )

    # plot history
    with plt.style.context('seaborn'):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].set_title('CTC')
        axes[0].plot(np.arange(NUM_EPOCHS) + 1, [ctc for ctc, _ in train_history], label='train')
        axes[0].plot(np.arange(NUM_EPOCHS) + 1, [ctc for ctc, _ in val_history], label='val')
        axes[0].set_xlabel('epoch'), axes[0].legend()
        axes[1].set_title('Levenshtein')
        axes[1].plot(np.arange(NUM_EPOCHS) + 1, [lev for _, lev in train_history], label='train')
        axes[1].plot(np.arange(NUM_EPOCHS) + 1, [lev for _, lev in val_history], label='val')
        axes[1].set_xlabel('epoch'), axes[1].legend()
        plt.savefig(f'train_val_history.png')
