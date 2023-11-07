"""
Sample code to train a segmentation model in PyTorch.
Adapted from here:
    https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb
Prerequisites:
    pip install segmentation-models-pytorch albumentations
Run as:
    python affordance-pred/train.py
"""

import os
import os.path as osp
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split

# Set random seeds
np.random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)

# ===================== Loading data =====================

curr_dir = osp.dirname(osp.abspath(__file__))
data_dir = osp.join(curr_dir, 'data/contactdb')
save_dir = osp.join(curr_dir, 'save')
os.makedirs(save_dir, exist_ok=True)
objs = ["apple", "banana", "cup", "cell_phone", "door_knob", "flashlight", "hammer", "knife", "light_bulb", "mug",
        "pan", "scissors", "stapler", "teapot", "toothbrush", "toothpaste"]

# generate train, val, test splits
x_all_fpaths = []
y_all_fpaths = []
for obj in objs:
    x_dir = osp.join(data_dir, "input", obj)
    y_dir = osp.join(data_dir, "gt", obj)
    x_all_fpaths.extend([osp.join(x_dir, fname) for fname in sorted(os.listdir(x_dir))])
    y_all_fpaths.extend([osp.join(y_dir, fname) for fname in sorted(os.listdir(x_dir))])

x_train_fpaths, x_eval_fpaths, y_train_fpaths, y_eval_fpaths = train_test_split(x_all_fpaths, y_all_fpaths, test_size=0.4, train_size=0.6, random_state=1024, shuffle=True)
x_val_fpaths, x_test_fpaths, y_val_fpaths, y_test_fpaths = train_test_split(x_eval_fpaths, y_eval_fpaths, test_size=0.5, train_size=0.5, random_state=1024, shuffle=True)

# helper function for data visualization
def visualize(fname="tmp.png", **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(5*n, 5.2))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.tight_layout()
    plt.savefig(fname=fname)
    plt.close()

# ===================== Dataloader =====================

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['grasp']

    def __init__(
            self,
            images_fps,
            masks_fps,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.images_fps = images_fps
        self.masks_fps = masks_fps

        # convert str names to class values on masks
        # self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.class_values = [255]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. grasp)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.images_fps)


# # Lets look at data we have
#
# dataset = Dataset(x_train_fpaths, y_train_fpaths, classes=['grasp'])
#
# sample_dir = osp.join(save_dir, "gt/train")
# os.makedirs(sample_dir, exist_ok=True)
# for i in range(100):
#     image, mask = dataset[i] # get some sample
#     visualize(
#         fname=osp.join(sample_dir, "%d.png"%i),
#         image=image,
#         affordance=mask.squeeze()
# )

# ===================== Augmentations =====================

import albumentations as albu


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=128, min_width=128, always_apply=True, border_mode=0),
        albu.RandomCrop(height=128, width=128, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(128, 128)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

#### Visualize resulted augmented images and masks

augmented_dataset = Dataset(
    x_train_fpaths,
    y_train_fpaths,
    augmentation=get_training_augmentation(),
    classes=['grasp'],
)

# same image with different random transforms
for i in range(3):
    image, mask = augmented_dataset[1]
    visualize(fname=osp.join(save_dir, "sample_augmentations_%d.png"%i), image=image, mask=mask.squeeze(-1))

# ===================== Create Model and Train =====================

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['grasp']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = Dataset(
    x_train_fpaths,
    y_train_fpaths,
    # augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_val_fpaths,
    y_val_fpaths,
    # augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])

# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# train model for 20 epochs

max_score = 0

for i in range(0, 20):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, osp.join(save_dir, 'best_model.pth'))
        print('Best model saved!')

    if i == 15:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

# ===================== Test best saved model =====================

# load best saved checkpoint
best_model = torch.load(osp.join(save_dir, 'best_model.pth'))

# create test dataset
test_dataset = Dataset(
    x_test_fpaths,
    y_test_fpaths,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)

# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = test_epoch.run(test_dataloader)

# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    x_test_fpaths, y_test_fpaths,
    classes=CLASSES,
)

results_dir = osp.join(save_dir, "predictions/test")
os.makedirs(results_dir, exist_ok=True)
for i in range(100):
    n = np.random.choice(len(test_dataset))

    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]

    gt_mask = gt_mask.squeeze()

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    visualize(
        fname=osp.join(results_dir, "%d.png"%i),
        image=image_vis,
        ground_truth_mask=gt_mask,
        predicted_mask=pr_mask
    )
