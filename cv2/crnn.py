import torch
import torch.nn as nn
import torchvision.models as models


class FeatureExtractor(nn.Module):
    def __init__(self, input_size=(64, 320), output_len=50):  # (128, 640)
        # output_len is the size of time steps for the bi-lstm layer
        # https://theailearner.com/2019/05/29/creating-a-crnn-model-to-recognize-text-in-an-image-part-1/
        super().__init__()
        h, w = input_size

        # resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # online
        # offline
        efficientnet = models.efficientnet_v2_s()  # offline
        state_dict = torch.load('weights/path/efficientnet_v2_s-dd5fe13b.pth')
        efficientnet.load_state_dict(state_dict)

        self.cnn = nn.Sequential(*list(efficientnet.children())[:-2])
        self.pool = nn.AvgPool2d(kernel_size=(h // 32, 1))
        self.proj = nn.Conv2d(w // 32, output_len, kernel_size=1)  # 320 // 32
        # in_channels, out_channels, kernel_size, stride=1,

        # self.num_output_features = self.cnn[-1][-1].bn2.num_features  # resnet18 512
        self.num_output_features = self.cnn[-1][-1][1].num_features  # efficientnet_v2_s 1280

    def apply_projection(self, x):
        """Use convolution to increase width of a features.
        Accepts tensor of features (shaped B x C x H x W).
        Returns new tensor of features (shaped B x C x H x W').
        """
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def forward(self, x):
        # Apply conv layers
        features = self.cnn(x)
        # Pool to make height == 1
        features = self.pool(features)
        # Apply projection to increase width
        features = self.apply_projection(features)
        return features
