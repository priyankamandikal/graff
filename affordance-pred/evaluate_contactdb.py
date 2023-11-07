"""
Evaluate affordance model on the contactdb dataset.
Taken from here:
    https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb
Prerequisites:
    pip install segmentation-models-pytorch albumentations
Run as:
    python affordance-pred/evaluate_contactdb.py
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
dataset_name = "contactdb"
data_dir = osp.join(curr_dir, 'data', dataset_name)
save_dir = osp.join(curr_dir, 'save')
os.makedirs(save_dir, exist_ok=True)

objs = ["apple", "cell_phone", "cup", "door_knob", "flashlight", "hammer", "knife", "light_bulb",
        "mouse", "mug", "pan", "scissors", "stapler", "teapot", "toothbrush", "toothpaste"]

# generate test paths
x_test_fpaths = {}
y_test_fpaths = {}
for obj in objs:
    x_test_fpaths[obj] = []
    y_test_fpaths[obj] = []
    x_dir = osp.join(data_dir, "input", obj)
    y_dir = osp.join(data_dir, "gt", obj)
    x_test_fpaths[obj].extend([osp.join(x_dir, fname) for fname in sorted(os.listdir(x_dir))])
    y_test_fpaths[obj].extend([osp.join(y_dir, fname) for fname in sorted(os.listdir(y_dir))])

# ===================== helper functions for data visualization =====================

# visualize image and affordance separately
def visualize_separate(fname="tmp.png", **images):
    """Plot image and affordance separately."""
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

# visualize affordance as hotspot on top of image
import cv2
def visualize_hotspots(fname, image, mask):
    """Plot affordance as hotspot on top of image."""
    mask = mask.astype(np.uint8)
    mask_rgba = cv2.merge((mask*0, mask*255, mask*0, mask*255))
    mask_rgba_blur = cv2.blur(mask_rgba, (3,3))
    image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    result = cv2.addWeighted(image_rgba, 1.0, mask_rgba_blur, 0.5, 0)
    cv2.imwrite(fname, result)

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
        # print(i, self.images_fps[i], self.masks_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        # print(image.shape, mask.shape)

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


# ===================== Preprocessing =====================

import albumentations as albu


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

# ===================== Load Model =====================

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

# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

# ===================== Test best saved model =====================

# load best saved checkpoint
best_model = torch.load(osp.join(save_dir, 'best_model.pth'))

for obj in objs:

    print('\n', '='*20, obj, '='*20)

    # create test dataset
    test_dataset = Dataset(
        x_test_fpaths[obj],
        y_test_fpaths[obj],
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

    # logs = test_epoch.run(test_dataloader)

    # test dataset without transformations for image visualization
    test_dataset_vis = Dataset(
        x_test_fpaths[obj],
        y_test_fpaths[obj],
        classes=CLASSES,
    )

    results_dir = osp.join(save_dir, "predictions/%s/%s"%(dataset_name,obj))
    os.makedirs(results_dir, exist_ok=True)
    for i in range(100):
        n = np.random.choice(len(test_dataset))

        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, _ = test_dataset[n]

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        # visualize_separate(
        #     fname=osp.join(results_dir, "%d.png"%i),
        #     image=image_vis,
        #     predicted_mask=pr_mask
        # )

        visualize_hotspots(
            fname=osp.join(results_dir, "%d.png" % i),
            image=image_vis,
            mask=pr_mask
        )


