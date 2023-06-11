from pathlib import Path
from collections import defaultdict
# from ast import literal_eval as make_tuple

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def show_25_labeled_images(data_path: Path, data_df: pd.DataFrame, first_idx: int = 0, save=False) -> None:
    if not first_idx % 25 == 0:
        raise AssertionError('first_idx % 25 should be equal to zero')
    plt.figure(figsize=(14, 12))
    for i, (img_name, lbl) in enumerate(data_df[first_idx:first_idx+25].to_numpy()):
        img = plt.imread(data_path / img_name)
        plt.subplot(5, 5, i + 1)
        plt.imshow(img)
        plt.grid(False)
        plt.title(f'{img_name}, {lbl}')
        plt.axis('off')
    if save:
        plt.savefig(f'{first_idx+1}_{first_idx+25}_labeled_images.png')
    plt.show()


def obtain_labeled_images_stats_inplace(labeled_data_path: Path, labels_df: pd.DataFrame, plot: bool = False) -> None:
    # labels_df: ['Id'] ['Expected']
    print('>>> calculate labeled data stats inplace')
    print(f'labeled samples: {labels_df.shape[0]}')
    labels_df.columns = ['img', 'text']  # Id,	Expected -> img, text
    print(labels_df.head())
    # drop bad labels
    print(f"drop {labels_df['text'].isna().sum()} NaN samples")
    labels_df.dropna(inplace=True)
    labels_df.reset_index(drop=True, inplace=True)
    print(f'labeled samples: {labels_df.shape[0]}')
    if plot:
        show_25_labeled_images(labeled_data_path, labels_df, first_idx=0, save=True)
        show_25_labeled_images(labeled_data_path, labels_df, first_idx=25, save=True)
    # obtain image stats
    labeled_files = [labeled_data_path / img_name for img_name in labels_df['img']]
    img_shapes = [plt.imread(p).shape for p in tqdm(labeled_files, total=len(labeled_files), desc='read files')]
    # plt.imread(p).shape -> height, width, num_channels
    labels_df['img_width'] = [s[1] for s in img_shapes]
    labels_df['img_height'] = [s[0] for s in img_shapes]
    labels_df['img_channels'] = [s[2] for s in img_shapes]
    labels_df['img_ratio'] = labels_df['img_width'] / labels_df['img_height']
    labels_df['text_len'] = labels_df['text'].apply(len)  # text label length
    # labels_df: ['img'] ['text'] ['img_width'] ['img_height'] ['img_channels'] ['img_ratio'] ['text_len']
    print(labels_df.head(5))


def calc_images_hists(labels_df: pd.DataFrame, columns: list, save_filename: str = None) -> None:
    # labels_df: ['img_width'] ['img_height'] ['img_channels'] ['img_ratio'] ['text_len']
    with plt.style.context('seaborn'):
        # columns = ['img_width', 'img_height', 'img_channels', 'img_ratio', 'text_len']
        fig, axes = plt.subplots(1, len(columns), figsize=(18, 3))
        for ax, col in zip(axes, columns):
            ax.set_title(col)
            ax.hist(labels_df[col], alpha=0.6, ec='white')
            ax.text(x=0.98, y=0.98, s=labels_df[col].describe().to_string(),
                    horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
            ax.set_yscale('log')
        if save_filename is not None:
            plt.savefig(f'{save_filename}.png')
        plt.show()


def delete_small_images_samples_inplace(labels_df: pd.DataFrame, img_sz_thresh: int = 5) -> None:
    # labels_df: ['img_width'] ['img_height']
    print('>>> deleting samples with small images inplace')
    small_imgs_df = labels_df[(labels_df.img_width <= img_sz_thresh) | (labels_df.img_height <= img_sz_thresh)]
    print(f'drop {small_imgs_df.shape[0]} samples (thresh={img_sz_thresh})')
    labels_df.drop(small_imgs_df.index, inplace=True)
    labels_df.reset_index(drop=True, inplace=True)
    print(f'labeled samples: {labels_df.shape[0]}')


def get_symbols_hist(labels_df: pd.DataFrame) -> list[tuple[str, int]]:
    # labels_df: ['text']
    symbols = defaultdict(int)
    for lbl in labels_df['text']:
        for c in lbl:
            symbols[c] += 1
    sorted_symbols_hist = sorted(symbols.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_symbols_hist


def delete_rare_symbols_samples_inplace(labels_df: pd.DataFrame, sym_freq_thresh: int = 3) -> None:
    # labels_df: ['text']
    print('>>> deleting samples with rare characters inplace')
    sorted_symbols = get_symbols_hist(labels_df)
    print(f'unique symbols: {len(sorted_symbols)}')
    print(f'first 30 symbols: {sorted_symbols[:30]}')
    # cropped_symbols = [(char, freq) for char, freq in sorted_symbols if freq > sym_freq_thresh]
    deleted_symbols = [char for char, freq in sorted_symbols if freq <= sym_freq_thresh]
    print(f'deleted symbols (thresh={sym_freq_thresh}): {len(deleted_symbols)}')
    bad_samples_df = labels_df[labels_df['text'].str.contains('|'.join(deleted_symbols))]
    # | if a string contains one of the substrings in a list
    print(f'drop {bad_samples_df.shape[0]} samples (thresh={sym_freq_thresh})')
    labels_df.drop(bad_samples_df.index, inplace=True)
    labels_df.reset_index(drop=True, inplace=True)
    print(f'labeled samples: {labels_df.shape[0]}')


def get_test_df(test_data_path: Path) -> pd.DataFrame:
    print('>>> calculate test data stats')
    test_files = sorted(test_data_path.glob('*'), key=lambda x: int(x.stem))  # sort by number
    # obtain image stats
    # test_files = [test_data_path / img_name for img_name in submission_df['img']]
    img_shapes = [plt.imread(p).shape for p in tqdm(test_files, total=len(test_files), desc='read files')]
    # plt.imread(p).shape -> height, width, num_channels
    test_df = pd.DataFrame()
    test_df['img'] = [p.name for p in test_files]
    test_df['img_width'] = [s[1] for s in img_shapes]
    test_df['img_height'] = [s[0] for s in img_shapes]
    test_df['img_channels'] = [s[2] for s in img_shapes]
    test_df['img_ratio'] = test_df['img_width'] / test_df['img_height']
    # test_df: ['img'] ['img_width'] ['img_height'] ['img_channels'] ['img_ratio']
    print(f'test samples: {test_df.shape[0]}')
    return test_df


def show_64_test_images(test_data_path: Path, test_df: pd.DataFrame, first_idx: int = 0, save=False) -> None:
    if not first_idx % 64 == 0:
        raise AssertionError('first_idx % 64 should be equal to zero')
    plt.figure(figsize=(14, 14))
    for i, img_name in enumerate(test_df.img[first_idx:first_idx+64]):
        img = plt.imread(test_data_path / img_name)
        plt.subplot(8, 8, i + 1)
        plt.imshow(img)
        plt.title(img_name)
        plt.grid(False)
        plt.axis('off')
    if save:
        plt.savefig(f'{first_idx+1}_{first_idx+64}_test_images.png')
    plt.show()
