from ultralytics.data.augment import Albumentations
from ultralytics.utils import LOGGER, colorstr
from ultralytics import YOLO
import cv2

def __init__(self, p=1.0):
    self.p = p
    self.transform = None
    prefix = colorstr("albumentations: ")

    try:
        import albumentations as A

        spatial_transforms = {
            "Affine",
            "BBoxSafeRandomCrop",
            "CenterCrop",
            "CoarseDropout",
            "Crop",
            "CropAndPad",
            "CropNonEmptyMaskIfExists",
            "D4",
            "ElasticTransform",
            "Flip",
            "GridDistortion",
            "GridDropout",
            "HorizontalFlip",
            "Lambda",
            "LongestMaxSize",
            "MaskDropout",
            "MixUp",
            "Morphological",
            "NoOp",
            "OpticalDistortion",
            "PadIfNeeded",
            "Perspective",
            "PiecewiseAffine",
            "PixelDropout",
            "RandomCrop",
            "RandomCropFromBorders",
            "RandomGridShuffle",
            "RandomResizedCrop",
            "RandomRotate90",
            "RandomScale",
            "RandomSizedBBoxSafeCrop",
            "RandomSizedCrop",
            "Resize",
            "Rotate",
            "SafeRotate",
            "ShiftScaleRotate",
            "SmallestMaxSize",
            "Transpose",
            "VerticalFlip",
            "XYMasking",
        }

        T = [
            A.Blur(p=0.2),
            A.MedianBlur(p=0.2),
            A.CLAHE(p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.3, 0.3), p=0.4),
        ]

        self.contains_spatial = any(
            transform.__class__.__name__ in spatial_transforms for transform in T)
        self.transform = (
            A.Compose(
                T,
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
            if self.contains_spatial
            else A.Compose(T)
        )
        LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
    except ImportError:
        pass
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")

if __name__ == '__main__':
    Albumentations.__init__ = __init__
    model = YOLO("yolov8n.pt")
    model.to('cuda')

    model.train(
                data="data.yaml",
                epochs=100,
                batch=8,
                imgsz=640,
                patience=10,
                plots=True,
                amp=False,
                )
    
    model.val(conf=0.5, iou=0.5)