from pathlib import Path

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class RecognitionDataset(Dataset):
    def __init__(self, files_dir, data_df, alphabet, transforms,
                 split='train', train_size=0.8, rotate_4=False):
        super().__init__()
        self.files_dir = Path(files_dir)
        self.alphabet = alphabet
        self.alphabet_dict = {c: i + 1 for i, c in enumerate(alphabet)}
        self.transforms = transforms
        self.split = split
        self.train_size = train_size
        self.rotate_4 = rotate_4

        if split in ['train', 'val']:
            img_filenames = data_df['img'].values
            train_val_border = int(len(img_filenames) * train_size) + 1
            first, last = (0, train_val_border) if split == 'train' else (train_val_border, len(img_filenames))
            self.filenames = img_filenames[first:last]
            img_labels = data_df['text'].values
            self.labels = img_labels[first:last]
            return
        if split == 'test':
            self.filenames = data_df['img'].values
            return
        raise NotImplementedError(f'Unknown split: {split}')

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = cv2.imread(str(self.files_dir / filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.split == 'test':
            # output = dict(image=image, filename=filename)
            output = dict(image=image, seq=[], seq_len=0, text='', filename=filename)
        else:  # 'train', 'val'
            text = self.labels[idx]
            seq = self._text_to_seq(text)
            output = dict(image=image, seq=seq, seq_len=len(seq), text=text, filename=filename)
        if self.rotate_4:  # bruteforce all possible text orientations
            # flip horizontal, flip vertical ??? (mirrored images)
            image_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            image_180 = cv2.rotate(image_90, cv2.ROTATE_90_CLOCKWISE)
            image_270 = cv2.rotate(image_180, cv2.ROTATE_90_CLOCKWISE)
            output['image'] = [image, image_90, image_180, image_270]
        if self.transforms is not None and self.rotate_4:
            for i in range(4):  # 4 possible image orientations
                output['image'][i] = self.transforms(output['image'][i])
        elif self.transforms is not None:
            output['image'] = self.transforms(output['image'])
        return output

    def _text_to_seq(self, text):
        # encode text to sequence of integers
        # Returns list of integers where each number is index of corresponding character in alphabet + 1
        # seq = [self.alphabet.find(c) + 1 for c in text]  # 0 -> blank
        seq = [self.alphabet_dict[c] for c in text]  # 0 -> blank
        return seq

    def __len__(self):
        return len(self.filenames)


class Cv2Resize:
    def __init__(self, output_size=(320, 64)):
        self.output_size = output_size

    def __call__(self, img):
        h, w = img.shape[:2]
        if w > self.output_size[0]:
            img = cv2.resize(img, self.output_size, interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, self.output_size, interpolation=cv2.INTER_CUBIC)
        return img.astype(np.uint8)


def collate_fn_recognition(batch: list[dict]) -> dict:
    """Function for torch.utils.data.Dataloader for batch collecting.
    Accepts list of dataset __get_item__ return values (dicts).
    Returns dict with same keys but values are either torch.Tensors of batched
    images, sequences, and so.
    """
    images, seqs, seq_lens, texts, file_names = [], [], [], [], []
    for sample in batch:
        images.append(sample['image'])
        seqs.extend(sample['seq'])
        seq_lens.append(sample['seq_len'])
        texts.append(sample['text'])
        file_names.append(sample['filename'])
    images = torch.stack(images)
    seqs = torch.Tensor(seqs).int()
    seq_lens = torch.Tensor(seq_lens).int()
    batch = {'image': images, 'seq': seqs, 'seq_len': seq_lens,
             'text': texts, 'file_name': file_names}
    return batch
