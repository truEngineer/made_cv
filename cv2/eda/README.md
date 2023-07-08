# Labeled images

First 25 image samples.

![labeled](/cv2/eda/25_labeled_images.png)

Histograms of all labeled images (`img_width`, `img_height`, `img_channels`, `img_ratio = img_width / img_height`, `text_len`).

![hists1](/cv2/eda/images_hists.png)

Histograms after removing small images (`width_thresh = height_thresh = 5 pixels`) and images with rare characters (`freq_thresh = 3 samples`).
A symbol is considered rare if it occurs less frequently than in three images.

![hists2](/cv2/eda/images_hists_st5_ft3.png)

# Test images

First 64 samples.

![test](/cv2/eda/64_test_images.png)
