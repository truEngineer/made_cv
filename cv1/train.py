from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models
from torchvision import transforms
# import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

np.random.seed(0)
torch.manual_seed(0)


class ImagesDataset(Dataset):
    def __init__(self, data_dir, data_df, split='train',
                 transform=transforms.Compose([transforms.ToTensor()]), target_transform=None):
        self.split = split
        split_df = None
        if self.split == 'train':
            split_df = data_df[data_df['val'] == 0]
        elif self.split == 'val':
            split_df = data_df[data_df['val'] == 1]
        elif self.split == 'test':
            split_df = data_df

        self.data_dir = data_dir
        self.files = split_df.image_id.to_numpy()

        if self.split in ('train', 'val'):
            self.labels = split_df.num_label.to_numpy()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.data_dir / self.files[idx]
        image = Image.open(file_path).convert('RGB')

        sample = None
        if self.split in ('train', 'val'):
            label = self.labels[idx]
            if self.target_transform:
                label = self.target_transform(label)
            sample = self.transform(image), label
        elif self.split == 'test':
            sample = (self.transform(image),)

        return sample


class ScaleAndPadToSquare:
    def __init__(self, output_size=224):
        self.output_size = output_size

    def __call__(self, img):
        old_size = img.size  # old_size (width, height)
        ratio = float(self.output_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        # resize the input image
        img = img.resize(new_size, Image.ANTIALIAS)
        # img = img.resize(new_size, Image.Resampling.LANCZOS)
        # create a new image and paste the resized on it
        new_img = Image.new('RGB', (self.output_size, self.output_size))
        new_img.paste(
            img, ((self.output_size - new_size[0]) // 2, (self.output_size - new_size[1]) // 2)
        )
        return new_img


class CropCenterSquare:
    def __init__(self, output_size=224):
        self.output_size = output_size

    def __call__(self, img):
        width, height = img.size
        new_width = min(width, height)
        new_height = new_width
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = (width + new_width) // 2
        bottom = (height + new_height) // 2
        # crop the center of the image
        img = img.crop((left, top, right, bottom))
        # resize image
        img = img.resize((self.output_size, self.output_size), Image.ANTIALIAS)
        # img = img.resize(new_size, Image.Resampling.LANCZOS)
        return img


def train(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    train_loss = []
    train_f1_score = []
    for batch in tqdm(loader, total=len(loader), desc="training...", position=0, leave=True):
        images = batch[0].to(device)  # B x 3 x SQUARE_SIZE x SQUARE_SIZE
        labels = batch[1].to(device)  # B x NUM_CLASSES
        with autocast():
            pred_logits = model(images)  # B x NUM_CLASSES
            loss = criterion(pred_logits, labels)  # loss(input, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # https://discuss.pytorch.org/t/should-it-really-be-necessary-to-do-var-detach-cpu-numpy/35489
        pred_labels = torch.argmax(pred_logits.detach(), dim=1).cpu().numpy()
        # my_tensor.detach().numpy(): to do some non-tracked computations based on the value of this tensor in a numpy array
        f1 = f1_score(y_true=labels.detach().cpu().numpy(), y_pred=pred_labels, average='micro')
        # can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

        train_loss.append(loss.item())
        train_f1_score.append(f1)
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

    return np.mean(train_loss), np.mean(train_f1_score)


def train_annealing(model, loader, criterion, optimizer, device, scheduler, epoch):
    model.train()
    iters = len(loader)
    i = 0
    train_loss = []
    for batch in tqdm(loader, total=len(loader), desc="training...", position=0, leave=True):
        images = batch[0].to(device)
        labels = batch[1].to(device)
        with autocast():
            pred_labels = model(images)
            loss = criterion(pred_labels, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss.append(loss.item())
        optimizer.zero_grad()

        scheduler.step(epoch + i / iters)
        i += 1

    return np.mean(train_loss)


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = []
    val_f1_score = []
    for batch in tqdm(loader, total=len(loader), desc="validation...", position=0, leave=True):
        images = batch[0].to(device)
        labels = batch[1].to(device)
        with torch.no_grad():
            pred_logits = model(images)
            loss = criterion(pred_logits, labels)  # loss(input, target)
            pred_labels = torch.argmax(pred_logits, dim=1).cpu().numpy()
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
            f1 = f1_score(y_true=labels.cpu().numpy(), y_pred=pred_labels, average='micro')

        val_loss.append(loss.item())
        val_f1_score.append(f1)
    return np.mean(val_loss), np.mean(val_f1_score)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    DATA_PATH = Path('./vk-made-sports-image-classification')

    train_df = pd.read_csv(DATA_PATH / 'train_fix270.csv')
    test_df = pd.read_csv(DATA_PATH / 'test.csv')
    labels = train_df.label.value_counts().index.tolist()
    label2number = {k: v for v, k in enumerate(labels)}
    number2label = {v: k for k, v in label2number.items()}

    print(f"train DF samples: {train_df.shape[0]}")
    # delete bad labels from train !!!
    # wrong_df = pd.read_csv('train_wrong2.csv')  # efficientnet_v2_s train predictions
    # train_df = train_df[~train_df['image_id'].isin(wrong_df['image_id'])]
    # print(f"remaining train samples: {train_df.shape[0]}")

    NUM_CLASSES = len(label2number)
    SAVE_PATH = Path("./models")
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    SQUARE_SIZE = 360  # 224 320 360 480
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    MODEL_NAME = "efficientnet_v2_s"
    # "densenet169" "densenet161" "efficientnet_b3" "efficientnet_v2_s" "efficientnet_v2_m"
    TRAIN_SIZE = 0.85
    NUM_EPOCHS = 20  # 20 30 40
    BATCH_SIZE = 32  # 32 64 128 256
    LEARNING_RATE = 1e-4  # 1e-3 5e-4 3e-4 1e-4
    TRANSFER = True

    train_files = [DATA_PATH / 'train' / id for id in train_df.image_id]
    train_df['num_label'] = train_df.label.map(label2number)
    # train_df['img_size'] = [Image.open(p).size for p in train_files]
    # train_df['img_ratio'] = train_df.img_size.apply(lambda x: x[0]) / train_df.img_size.apply(lambda x: x[1])  # width/height
    # train_df['img_mode'] = [Image.open(p).mode for p in train_files]

    test_files = [DATA_PATH / 'test' / id for id in test_df.image_id]
    # test_df['img_size'] = [Image.open(p).size for p in test_files]
    # test_df['img_ratio'] = test_df.img_size.apply(lambda x: x[0]) / test_df.img_size.apply(lambda x: x[1])  # width/height
    # test_df['img_mode'] = [Image.open(p).mode for p in test_files]

    # Train/val split inside train_df
    # np.random.rand: random samples from a uniform distribution over [0, 1)
    train_df['val'] = (np.random.rand(len(train_df)) > TRAIN_SIZE) * 1
    print(f"validation samples count: {train_df['val'].sum()}")
    print(train_df.loc[train_df.val == 1, 'label'].value_counts())

    train_transforms = transforms.Compose([
        # transforms.Resize(224),  # Resize will behave differently on input images with a different height and width
        ScaleAndPadToSquare(SQUARE_SIZE),
        # CropCenterSquare(SQUARE_SIZE),
        transforms.RandomGrayscale(p=0.3),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.RandomApply(transforms=[transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.2, 2))], p=0.3),
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    test_transforms = transforms.Compose([
        ScaleAndPadToSquare(SQUARE_SIZE),
        # CropCenterSquare(SQUARE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    train_dataset = ImagesDataset(
        data_dir=DATA_PATH / 'train',
        data_df=train_df,
        split='train',
        transform=train_transforms
    )

    val_dataset = ImagesDataset(
        data_dir=DATA_PATH / 'train',
        data_df=train_df,
        split='val',
        transform=test_transforms
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        num_workers=4, pin_memory=True,
        shuffle=True, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        num_workers=4, pin_memory=True,
        shuffle=False, drop_last=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Creating model...")
    # densenet121   7 978 856 params
    # densenet169  14 149 480 params
    # densenet201  20 013 928 params
    # densenet161  28 681 000 params

    # efficientnet_b0   5 288 548 params
    # efficientnet_b1   7 794 184 params
    # efficientnet_b2   9 109 994 params
    # efficientnet_b3  12 233 232 params
    # efficientnet_b4  19 341 616 params
    # efficientnet_b5  30 389 784 params
    # efficientnet_b6  43 040 704 params
    # efficientnet_b7  66 347 960 params

    # efficientnet_v2_s   21 458 488 params
    # efficientnet_v2_m   54 139 356 params
    # efficientnet_v2_l  118 515 272 params

    model = None

    if MODEL_NAME == "densenet161":
        model = models.densenet161(weights='IMAGENET1K_V1')
        # (classifier): Linear(in_features=1664, out_features=1000, bias=True)
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES, bias=True)
    elif MODEL_NAME == "densenet169":
        model = models.densenet169(weights='IMAGENET1K_V1')
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES, bias=True)
    elif MODEL_NAME == "efficientnet_b3":
        model = models.efficientnet_b3()
        # (classifier): Sequential(
        #     (0): Dropout(p=0.3, inplace=True)
        #     (1): Linear(in_features=1536, out_features=1000, bias=True)
        # )
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(model.classifier[1].in_features, NUM_CLASSES, bias=True)
        )
    elif MODEL_NAME == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')  # models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        # (classifier): Sequential(
        #     (0): Dropout(p=0.2, inplace=True)
        #     (1): Linear(in_features=1280, out_features=1000, bias=True)
        # )
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(model.classifier[1].in_features, NUM_CLASSES, bias=True)
        )
    elif MODEL_NAME == "efficientnet_v2_m":
        model = models.efficientnet_v2_m(weights='IMAGENET1K_V1')  # models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        # (classifier): Sequential(
        #     (0): Dropout(p=0.3, inplace=True)
        #     (1): Linear(in_features=1280, out_features=1000, bias=True)
        # )
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(model.classifier[1].in_features, NUM_CLASSES, bias=True)
        )

    model.to(device)
    print(f"{count_parameters(model)} trainable params")
    if TRANSFER:
        for p in model.features.parameters():
            p.requires_grad = False

        # efficientnet_v2_s / efficientnet_v2_m
        for p in model.features[-2:].parameters():  # (6): Sequential, (7): Conv2dNormActivation
            p.requires_grad = True
        # for p in model.avgpool.parameters():
        #     p.requires_grad = True  # by default
        # for p in model.classifier.parameters():
        #     p.requires_grad = True  # by default
        print(f"Transfer learning: {count_parameters(model)} trainable params")

    scaler = GradScaler()  # to decrease GPU memory usage

    # n_samples / (n_classes * np.bincount(y))
    # https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
    class_weights = len(train_df) / (NUM_CLASSES * train_df.label.value_counts().to_numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction="mean")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, amsgrad=True, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS,
                                              max_lr=LEARNING_RATE, final_div_factor=1e3)
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True, weight_decay=0.00001)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader), T_mult=2)

    # print("Ready for training...")

    best_val_loss = np.inf

    for epoch in range(NUM_EPOCHS):
        # train_loss = train_annealing(model, train_loader, criterion, optimizer, device, scheduler, epoch)
        train_ce_loss, train_f1 = train(model, train_loader, criterion, optimizer, device, scheduler=scheduler)  # scheduler=None
        val_ce_loss, val_f1 = validate(model, val_loader, criterion, device=device)
        log_df = pd.DataFrame(
            {'train_ce': [train_ce_loss], 'val_ce': [val_ce_loss], 'train_f1': [train_f1], 'val_f1': [val_f1]}
        )
        if epoch == 0:
            log_df.to_csv(SAVE_PATH / f'{MODEL_NAME}_train_log.csv', mode='w', header=True, index=False)
        else:
            log_df.to_csv(SAVE_PATH / f'{MODEL_NAME}_train_log.csv', mode='a', header=False, index=False)
        print(f"Epoch #{epoch + 1:<{2}}")
        print(f"train CE loss: {train_ce_loss:.3f}  val CE loss: {val_ce_loss:.3f}")
        print(f"train F1: {train_f1:.3f}  val F1: {val_f1:.3f}")
        if val_ce_loss < best_val_loss:
            best_val_loss = val_ce_loss
            with open(SAVE_PATH / f"{MODEL_NAME}_ep{epoch+1}_t{train_ce_loss:.2f}_v{val_ce_loss:.2f}.pth", "wb") as fp:
                torch.save(model.state_dict(), fp)
