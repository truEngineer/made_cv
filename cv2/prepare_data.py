from pathlib import Path

import pandas as pd

from data_analysis import obtain_labeled_images_stats_inplace, calc_images_hists

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
    
