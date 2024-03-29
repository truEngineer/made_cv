import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
# from torchaudio.functional import edit_distance as distance
from Levenshtein import distance


def pred_to_string(pred: np.ndarray, alphabet: str) -> (str, float):
    seq = []
    log_probs = []
    for i in range(len(pred)):  # iterate CNN output 1 ... 30
        label = np.argmax(pred[i])  # greedy decode (alphabet size 234)
        seq.append(label - 1)  # 0: blank -> -1
        log_probs.append(pred[i][label])
    out_symbols = []
    out_log_probs = []
    for i in range(len(seq)):
        if len(out_symbols) == 0:
            if seq[i] != -1:  # skip blank
                out_symbols.append(seq[i])
                out_log_probs.append(log_probs[i])
        else:
            if seq[i] != -1 and seq[i] != seq[i - 1]:
                out_symbols.append(seq[i])
                out_log_probs.append(log_probs[i])
    out_str = ''.join([alphabet[c] for c in out_symbols])
    return out_str, sum(out_log_probs)


def decode(preds: torch.Tensor, alphabet: str) -> (list[str], list[float]):
    preds = preds.permute(1, 0, 2).cpu().data.numpy()  # torch.Size([30, 256, 234]) -> torch.Size([256, 30, 234])
    outputs, log_probs = [], []
    # preds = preds.permute(1, 0, 2).cpu()
    # preds = F.log_softmax(preds, dim=2).data.numpy()  # obtain log probs !!!
    for i in range(len(preds)):  # iterate through batch elements 1 ... 256
        # outputs.append(pred_to_string(preds[i], alphabet))  # old version, outputs only
        out_str, out_log_prob = pred_to_string(preds[i], alphabet)
        outputs.append(out_str)
        log_probs.append(out_log_prob)
    return outputs, log_probs


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_train_val_epoch(model: nn.Module, data_loader: DataLoader,
                          device: torch.device, optimizer: optim.Optimizer = None) -> tuple[list, list]:
    epoch_losses, levenshtein_losses = [], []

    mode = 'train' if optimizer is not None else 'val'

    if mode == 'train':
        model.train()
    elif mode == 'val':
        model.eval()

    sample_fname, sample_gt, sample_pred, sample_dist = [], [], [], []
    for i, b in tqdm(enumerate(data_loader), total=len(data_loader), desc=mode):
        images = b['image'].to(device)
        seqs_gt = b['seq']
        seq_lens_gt = b['seq_len']

        if mode == 'train':
            optimizer.zero_grad()
            seqs_pred = model(images).cpu()
        elif mode == 'val':
            with torch.no_grad():
                seqs_pred = model(images).cpu()

        log_probs = F.log_softmax(seqs_pred, dim=2)
        seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()
        # input_lengths = torch.IntTensor(batch_size).fill_(model.postconv_width)

        texts_pred, _ = decode(seqs_pred, model.alphabet)
        texts_gt = b['text']
        levenshtein_losses.extend([distance(pred, gt) for pred, gt in zip(texts_pred, texts_gt)])

        loss = F.ctc_loss(log_probs=log_probs,  # (T, N, C) or (T, C)
                          targets=seqs_gt,  # N, S or sum(target_lengths)
                          input_lengths=seq_lens_pred,  # N
                          target_lengths=seq_lens_gt,  # N
                          blank=0, reduction='mean', zero_infinity=True)
        # Whether to zero infinite losses and the associated gradients.
        # Default: False Infinite losses mainly occur when the inputs are too short to be aligned to the targets.

        if mode == 'train':
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if mode == 'val' and i == len(data_loader) - 1:  # print last samples
            sample_fname.extend(b['file_name'][:5])
            sample_gt.extend(texts_gt[:5])
            sample_pred.extend(texts_pred[:5])
            sample_dist.extend([distance(pred, gt) for pred, gt in zip(sample_gt, sample_pred)])

        epoch_losses.append(loss.item())

    # print last samples
    for fname, gt, pred, d in zip(sample_fname, sample_gt, sample_pred, sample_dist):
        print(f'{fname}, gt: "{gt}", pred: "{pred}", dist: {d}')

    return epoch_losses, levenshtein_losses


def train_model(model: nn.Module, experiment_name: str, num_epochs: int,
                device: torch.device, optimizer: optim.Optimizer, lr_scheduler: optim.lr_scheduler,
                train_dataloader: DataLoader, val_dataloader: DataLoader) -> tuple[list[tuple], list[tuple]]:
    best_loss = np.inf
    prev_lr = optimizer.param_groups[0]['lr']
    train_history, val_history = [], []

    for i in range(num_epochs):
        # Load best model if LR has been changed
        if optimizer.param_groups[0]['lr'] < prev_lr:
            prev_lr = optimizer.param_groups[0]['lr']
            with open(f'{experiment_name}.pth', 'rb') as fp:
                state_dict = torch.load(fp, map_location='cpu')
            model.load_state_dict(state_dict)
            model.to(device)
        # TRAINING
        epoch_losses, levenshtein_losses = model_train_val_epoch(model, train_dataloader, device, optimizer)
        print(f'epoch {i + 1}, ctc: {np.mean(epoch_losses):.3f}, levenshtein: {np.mean(levenshtein_losses):.3f}')
        train_history.append((np.mean(epoch_losses), np.mean(levenshtein_losses)))
        time.sleep(0.5)
        # VALIDATION
        epoch_losses, levenshtein_losses = model_train_val_epoch(model, val_dataloader, device)

        if best_loss > np.mean(epoch_losses):
            best_loss = np.mean(epoch_losses)
            with open(f'{experiment_name}.pth', 'wb') as fp:
                torch.save(model.state_dict(), fp)

        lr_scheduler.step(np.mean(levenshtein_losses))
        print(f'epoch {i + 1}, ctc: {np.mean(epoch_losses):.3f}, levenshtein: {np.mean(levenshtein_losses):.3f}')
        print(f'best val loss: {best_loss:.3f}\n')
        val_history.append((np.mean(epoch_losses), np.mean(levenshtein_losses)))
        time.sleep(0.5)

    return train_history, val_history


def test_model(model: nn.Module, experiment_name: str, device: torch.device, test_dataloader: DataLoader) -> list[str]:
    with open(f'{experiment_name}.pth', 'rb') as fp:
        state_dict = torch.load(fp, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    test_preds = []
    for b in tqdm(test_dataloader, total=len(test_dataloader), desc='test'):
        images = b['image'].to(device)
        with torch.no_grad():
            seqs_pred = model(images).cpu()
        texts_pred, _ = decode(seqs_pred, model.alphabet)
        test_preds.extend(texts_pred)
    return test_preds


def train_check_batch1_model(model: nn.Module, experiment_name: str, device: torch.device,
                             dataloader: DataLoader) -> pd.DataFrame:
    if not dataloader.batch_size == 1:  # collate_fn not needed!
        raise AssertionError('dataloader batch_size = 1 is required')
    with open(f'{experiment_name}.pth', 'rb') as fp:
        state_dict = torch.load(fp, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    img_filenames, img_texts, img_angle_preds_distances = [], [], []
    img_angles = ['img_0', 'img_90', 'img_180', 'img_270']
    for b in tqdm(dataloader, total=len(dataloader), desc='train check'):
        # print(len(b['image']), b['image'][0].shape, b['image'][1].shape, b['filename'], b['text'])
        # 4 torch.Size([1, 3, 64, 320]) torch.Size([1, 3, 64, 320]) ['1.jpg'] ['Атырау']
        img_filenames.append(b['filename'])
        img_texts.append(b['text'])
        images = torch.stack([img_t.squeeze(0) for img_t in b['image']]).to(device)  # bs = 1
        # b['image'] = [image_0, image_90, image_180, image_270]
        with torch.no_grad():
            seqs_pred = model(images).cpu()

        texts_pred, _ = decode(seqs_pred, model.alphabet)  # texts_gt: list[str]
        texts_gt = [b['text']] * len(b['image'])  # ['label'] * 4 = ['label', 'label', 'label', 'label']
        # len(b['image']) == len(img_angles)
        img_angle_preds_distances.append(
            {angle: (pred, distance(pred, gt)) for angle, pred, gt in zip(img_angles, texts_pred, texts_gt)}
        )  # list[dict[tuple]]
    rotation_df = pd.DataFrame()
    rotation_df['img'] = img_filenames
    rotation_df['text'] = img_texts
    for angle in img_angles:
        rotation_df[f'{angle}_pred'] = [img[angle][0] for img in img_angle_preds_distances]
        rotation_df[f'{angle}_dist'] = [img[angle][1] for img in img_angle_preds_distances]

    return rotation_df


def bruteforce_rotation_model(model: nn.Module, experiment_name: str, device: torch.device,
                              dataloader: DataLoader, labeled_data: bool = False) -> pd.DataFrame:
    with open(f'{experiment_name}.pth', 'rb') as fp:
        state_dict = torch.load(fp, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    img_filenames, img_angles, img_preds, img_log_probs = [], [], [], []
    img_texts, img_dists = [], []  # labeled_data=True
    angles = [0, 90, 180, 270]
    # dataset.RecognitionDataset(rotate=True)
    for b in tqdm(dataloader, total=len(dataloader), desc='train rotation check'):
        # print(len(b['image']), b['image'][0].shape, b['image'][1].shape, len(b['file_name']), len(b['text']))
        # 256 torch.Size([3, 64, 320]) torch.Size([3, 64, 320]) 256 256
        img_filenames.extend(b['file_name'])
        # dataloader image bs = 64, collate fn image bs = 64 * 4
        img_angles.extend(angles * (len(b['image']) // 4))  # current batch size = len(b['image']) // len(angles)
        images = b['image'].to(device)
        # b['image'] = [img_0_0, img_0_90, img_0_180, img_0_270, img_1_0, img_1_90, img_1_180, img_1_270, img_2_0, ...]
        with torch.no_grad():
            seqs_pred = model(images).cpu()

        # torch.Size([30, 256, 234])
        # log_probs = F.log_softmax(seqs_pred, dim=2)
        # torch.Size([30, 256, 234])
        texts_pred, log_probs = decode(seqs_pred, model.alphabet)
        img_preds.extend(texts_pred)
        img_log_probs.extend(log_probs)
        if labeled_data:
            img_texts.extend(b['text'])
            img_dists.extend(
                [distance(pred, gt) for pred, gt in zip(texts_pred, b['text'])]  # b['text'] = texts_gt: list[str]
            )

    rotation_df = pd.DataFrame()
    rotation_df['img'] = img_filenames
    rotation_df['angle'] = img_angles
    rotation_df['pred'] = img_preds
    if labeled_data:
        rotation_df['text'] = img_texts
        rotation_df['dist'] = img_dists
    rotation_df['logit'] = img_log_probs

    return rotation_df
