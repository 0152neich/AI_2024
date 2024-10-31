import cv2
from data_augmentation.augmentation import YOLODataAugmentation
from configs.config import *

if __name__ == "__main__":
    augmenter = YOLODataAugmentation(IMAGE_PATH + "1.png")
    augmented_images = augmenter.apply_all({
        "width": IMG_WIDTH,
        "height": IMG_HEIGHT,
        "angle": ROTATION_ANGLE,
        "blur_type": BLUR_TYPE,
        "brightness": BRIGHTNESS_JITTER,
        "contrast": CONTRAST_JITTER,
        "saturation": SATURATION_JITTER,
        "hue": HUE_JITTER,
        "noise_mean": NOISE_MEAN,
        "noise_sigma": NOISE_SIGMA,
        "scale_fx": SCALE_FX,
        "scale_fy": SCALE_FY,
        "x_shift": X_SHIFT,
        "y_shift": Y_SHIFT,
        "ksize": KSIZE,
        "crop_size": CROP_SIZE,
        "erase_area_ratio": ERASE_AREA_RATIO,
        "cutmix_alpha": CUTMIX_ALPHA,
        "mixup_alpha": MIXUP_ALPHA,
        "paste_size": PASTE_SIZE
    })

    for title, image in augmented_images.items():
        cv2.imshow(title, image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            print("Chương trình đã dừng.")
            break
    cv2.destroyAllWindows()