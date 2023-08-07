from pathlib import Path

import pandas as pd

from data_analysis import (obtain_labeled_images_stats_inplace,
                           delete_small_images_samples_inplace,
                           delete_rare_symbols_samples_inplace,
                           calc_images_hists, 
                           get_test_df, show_64_test_images)

IMG_SIZE_THRESH = 5
SYM_FREQ_THRESH = 3

np.random.seed(0)
torch.manual_seed(0)


if __name__ == '__main__':
    data_path = Path('./ocr_contest_data')
    labeled_data_path = Path(data_path / 'train')
    test_data_path = Path(data_path / 'test')
    
    # 1. calc image stats
    labels_df = pd.read_csv(data_path / 'train_labels.csv')
    obtain_labeled_images_stats_inplace(labeled_data_path, labels_df, plot=True)  # update labels_df inplace !!! time
    labels_df.to_csv(data_path / 'extended_train_labels.csv', header=True, index=False)
    columns = ['img_width', 'img_height', 'img_channels', 'img_ratio', 'text_len']
    calc_images_hists(labels_df, columns, save_filename='images_hists')

    # 2. deleting samples with small images
    labels_df = pd.read_csv(data_path / 'extended_train_labels.csv')
    delete_small_images_samples_inplace(labels_df, IMG_SIZE_THRESH)  # update labels_df inplace
    labels_df.to_csv(data_path / f'extended_train_labels_st{IMG_SIZE_THRESH}.csv',
                     header=True, index=False)  # st: size threshold
    calc_images_hists(labels_df, columns, save_filename=f'images_hists_st{IMG_SIZE_THRESH}')

    # 3. deleting samples with rare characters
    labels_df = pd.read_csv(data_path / f'extended_train_labels_st{IMG_SIZE_THRESH}.csv')
    delete_rare_symbols_samples_inplace(labels_df, SYM_FREQ_THRESH)  # update labels_df inplace
    labels_df.to_csv(data_path / f'extended_train_labels_st{IMG_SIZE_THRESH}_ft{SYM_FREQ_THRESH}.csv',
                     header=True, index=False)  # ft: freq threshold
    calc_images_hists(labels_df, columns, save_filename=f'images_hists_st{IMG_SIZE_THRESH}_ft{SYM_FREQ_THRESH}')

    # 4. test data analysis
    # print(sorted(test_data_path.glob('*'), key=lambda x: int(x.stem))[:100])
    test_df = get_test_df(test_data_path)  # !!! time
    test_df.to_csv(data_path / 'test_data.csv', header=True, index=False)
    # test_df = pd.read_csv(data_path / f'test_data.csv')
    show_64_test_images(test_data_path, test_df, first_idx=0, save=True)
    columns = ['img_width', 'img_height', 'img_channels', 'img_ratio']
    calc_images_hists(test_df, columns, save_filename='test_images_hists')
