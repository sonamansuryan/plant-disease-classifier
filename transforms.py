import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(8, img_size // 10),
            hole_width_range=(8, img_size // 10),
            p=0.3,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int) -> A.Compose:
    new_size = int(img_size * 1.14)
    return A.Compose([
        A.Resize(height=new_size, width=new_size),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
