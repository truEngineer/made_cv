import torch
import torch.nn as nn
import torchvision.models as models


class FeatureExtractor(nn.Module):
    def __init__(self, input_size=(64, 320), output_len=50):  # (128, 640)
        # output_len is the size of time steps for the bi-lstm layer
        # https://theailearner.com/2019/05/29/creating-a-crnn-model-to-recognize-text-in-an-image-part-1/
        super().__init__()
        h, w = input_size

        # online
        # resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        efficientnet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        
        # offline
        # efficientnet = models.efficientnet_v2_s()
        # state_dict = torch.load('weights/path/efficientnet_v2_s-dd5fe13b.pth')
        # efficientnet.load_state_dict(state_dict)

        self.cnn = nn.Sequential(*list(efficientnet.children())[:-2])  # resnet.children()
        self.pool = nn.AvgPool2d(kernel_size=(h // 32, 1))
        self.proj = nn.Conv2d(w // 32, output_len, kernel_size=1)  # 320 // 32
        # in_channels, out_channels, kernel_size, stride=1,

        # self.num_output_features = self.cnn[-1][-1].bn2.num_features  # resnet18 512
        self.num_output_features = self.cnn[-1][-1][1].num_features     # efficientnet_v2_s 1280

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


class SequencePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 dropout=0.3, bidirectional=False):
        super().__init__()
        self.num_classes = num_classes
        self.rnn = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout,
            bidirectional=bidirectional
        )
        fc_in = hidden_size if not bidirectional else 2 * hidden_size
        self.fc = nn.Linear(in_features=fc_in, out_features=num_classes)

    def _init_hidden_(self, batch_size):
        """Initialize new tensor of zeroes for RNN hidden state.
        Accepts batch size.
        Returns tensor of zeros shaped
        (num_layers * num_directions, batch, hidden_size).
        """
        num_directions = 2 if self.rnn.bidirectional else 1
        return torch.zeros(self.rnn.num_layers * num_directions,
                           batch_size, self.rnn.hidden_size)

    def _prepare_features_(self, x):
        """Change dimensions of x to fit RNN expected input.
        Accepts tensor x shaped (B x (C=1) x H x W).
        Returns new tensor shaped (W x B x H).
        """
        x = x.squeeze(1)
        x = x.permute(2, 0, 1)
        return x

    def forward(self, x):
        x = self._prepare_features_(x)
        batch_size = x.size(1)
        h_0 = self._init_hidden_(batch_size)
        h_0 = h_0.to(x.device)
        x, h = self.rnn(x, h_0)
        x = self.fc(x)
        return x
