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
