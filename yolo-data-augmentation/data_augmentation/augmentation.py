import cv2
import numpy as np
import random
from typing import Dict, Optional, Tuple

class YOLODataAugmentation:
    def __init__(self, img_path: str = "") -> None:
        """Initialize YOLODataAugmentation with an image path.

        Args:
            img_path (str, optional): Path to the image file. Defaults to "".

        Raises:
            ValueError: If the image cannot be loaded from the provided path.
        """
        self.img: Optional[np.ndarray] = cv2.imread(img_path)
        if self.img is None:
            raise ValueError(f"Image at path '{img_path}' cannot be loaded. Check the path and try again.")

    def resize_img(self, img: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize the image to the specified width and height.

        Args:
            img (np.ndarray): Input image to resize.
            width (int): Target width.
            height (int): Target height.

        Returns:
            np.ndarray: Resized image.
        """
        return cv2.resize(img, (width, height))

    def flip_horizontal(self, img: np.ndarray) -> np.ndarray:
        """Flip the image horizontally.

        Args:
            img (np.ndarray): Input image to flip.

        Returns:
            np.ndarray: Horizontally flipped image.
        """
        return cv2.flip(img, 1)

    def rotate_img(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Rotate the image by a specified angle.

        Args:
            img (np.ndarray): Input image to rotate.
            angle (float): Angle of rotation in degrees.

        Returns:
            np.ndarray: Rotated image.
        """
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        return cv2.warpAffine(img, rotation_matrix, (w, h))

    def add_color_jitter(self, img: np.ndarray, brightness: float = 0.3, contrast: float = 0.3,
                         saturation: float = 0.3, hue: float = 0.3) -> np.ndarray:
        """Apply color jitter to the image.

        Args:
            img (np.ndarray): Input image to apply color jitter.
            brightness (float, optional): Brightness adjustment factor. Defaults to 0.3.
            contrast (float, optional): Contrast adjustment factor. Defaults to 0.3.
            saturation (float, optional): Saturation adjustment factor. Defaults to 0.3.
            hue (float, optional): Hue adjustment factor. Defaults to 0.3.

        Returns:
            np.ndarray: Image with color jitter applied.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 2] *= 1 + random.uniform(-brightness, brightness)
        hsv[..., 1] *= 1 + random.uniform(-saturation, saturation)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def add_noise(self, img: np.ndarray, mean: float = 0, sigma: float = 25) -> np.ndarray:
        """Add Gaussian noise to the image.

        Args:
            img (np.ndarray): Input image to add noise.
            mean (float, optional): Mean of Gaussian noise. Defaults to 0.
            sigma (float, optional): Standard deviation of Gaussian noise. Defaults to 25.

        Returns:
            np.ndarray: Image with noise added.
        """
        noise = np.random.normal(mean, sigma, img.shape).astype(np.uint8)
        return cv2.add(img, noise)

    def mosaic(self, images: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
               size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """Create a mosaic of four images.

        Args:
            images (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): A tuple of four images.
            size (Tuple[int, int], optional): Output size of the mosaic image. Defaults to (640, 640).

        Returns:
            np.ndarray: Mosaic image.
        """
        assert len(images) == 4, "Mosaic requires exactly 4 images."
        h, w = size[0] // 2, size[1] // 2
        images_resized = [cv2.resize(img, (w, h)) for img in images]
        top = np.hstack(images_resized[:2])
        bottom = np.hstack(images_resized[2:])
        return np.vstack((top, bottom))

    def scale(self, img: np.ndarray, fx: float = 1.2, fy: float = 1.2) -> np.ndarray:
        """Scale the image by the specified factors.

        Args:
            img (np.ndarray): Input image to scale.
            fx (float, optional): Scaling factor along the x-axis. Defaults to 1.2.
            fy (float, optional): Scaling factor along the y-axis. Defaults to 1.2.

        Returns:
            np.ndarray: Scaled image.
        """
        return cv2.resize(img, None, fx=fx, fy=fy)

    def translate(self, img: np.ndarray, x_shift: float = 50, y_shift: float = 50) -> np.ndarray:
        """Translate the image by specified shifts.

        Args:
            img (np.ndarray): Input image to translate.
            x_shift (float, optional): Shift along the x-axis. Defaults to 50.
            y_shift (float, optional): Shift along the y-axis. Defaults to 50.

        Returns:
            np.ndarray: Translated image.
        """
        (h, w) = img.shape[:2]
        translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        return cv2.warpAffine(img, translation_matrix, (w, h))

    def apply_blur(self, img: np.ndarray, blur_type: str = 'gaussian', ksize: int = 5) -> np.ndarray:
        """Apply blur to the image based on the specified blur type.

        Args:
            img (np.ndarray): Input image to blur.
            blur_type (str, optional): Type of blur ('gaussian', 'median', or 'bilateral'). Defaults to 'gaussian'.
            ksize (int, optional): Kernel size for the blur. Defaults to 5.

        Returns:
            np.ndarray: Blurred image.
        """
        if blur_type == 'gaussian':
            return cv2.GaussianBlur(img, (ksize, ksize), 0)
        elif blur_type == 'median':
            return cv2.medianBlur(img, ksize)
        elif blur_type == 'bilateral':
            return cv2.bilateralFilter(img, ksize, 75, 75)
    
    def random_crop(self, img: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
        """Randomly crop a portion of the image.

        Args:
            img (np.ndarray): Input image to crop.
            crop_size (Tuple[int, int]): Size of the crop (height, width).

        Returns:
            np.ndarray: Cropped image.
        """
        h, w = img.shape[:2]
        ch, cw = crop_size
        x = random.randint(0, w - cw)
        y = random.randint(0, h - ch)
        return img[y:y + ch, x:x + cw]

    def random_erasing(self, img: np.ndarray, area_ratio: float = 0.2) -> np.ndarray:
        """Randomly erase a portion of the image.

        Args:
            img (np.ndarray): Input image to apply random erasing.
            area_ratio (float, optional): Ratio of area to erase. Defaults to 0.2.

        Returns:
            np.ndarray: Image with a portion erased.
        """
        h, w = img.shape[:2]
        erase_area = int(h * w * area_ratio)
        erase_h = random.randint(1, h)
        erase_w = erase_area // erase_h
        if erase_w >= w or erase_h >= h:
            print("Erased dimensions exceed image dimensions. Skipping random erasing.")
            return img
        x = random.randint(0, w - erase_w)
        y = random.randint(0, h - erase_h)
        
        img[y:y + erase_h, x:x + erase_w] = 0  # Erasing with a black rectangle
        return img

    def cutmix(self, img1: np.ndarray, img2: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Apply CutMix augmentation.

        Args:
            img1 (np.ndarray): First input image.
            img2 (np.ndarray): Second input image.
            alpha (float, optional): Parameter for Beta distribution. Defaults to 1.0.

        Returns:
            np.ndarray: Mixed image.
        """
        lam = np.random.beta(alpha, alpha)
        h, w, _ = img1.shape
        cut_h = int(h * lam)
        cut_w = int(w * lam)
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        x1 = np.clip(cx - cut_w // 2, 0, w)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        y2 = np.clip(cy + cut_h // 2, 0, h)

        img1[y1:y2, x1:x2] = img2[y1:y2, x1:x2]  # Mix the images
        return img1
    
    def mixup(self, img1: np.ndarray, img2: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Apply MixUp augmentation.

        Args:
            img1 (np.ndarray): First input image.
            img2 (np.ndarray): Second input image.
            alpha (float, optional): Parameter for Beta distribution. Defaults to 1.0.

        Returns:
            np.ndarray: Mixed image.
        """
        lam = np.random.beta(alpha, alpha)
        return (img1 * lam + img2 * (1 - lam)).astype(np.uint8)
    
    def copy_paste(self, img1: np.ndarray, img2: np.ndarray, paste_size: tuple = (100, 100)) -> np.ndarray:
        """Apply Copy-Paste augmentation.

        Args:
            img1 (np.ndarray): The original input image.
            img2 (np.ndarray): The image to paste from.
            paste_size (tuple, optional): The size of the area to paste. Defaults to (100, 100).

        Returns:
            np.ndarray: Image after applying Copy-Paste augmentation.
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        x_offset = random.randint(0, w1 - paste_size[0])
        y_offset = random.randint(0, h1 - paste_size[1])
        
        img2_resized = cv2.resize(img2, paste_size)
        
        img1[y_offset:y_offset + paste_size[1], x_offset:x_offset + paste_size[0]] = img2_resized
        
        return img1

    
    def apply_all(self, params=None):
        """Apply all augmentation methods to the original image.

        Args:
            params (dict, optional): A dictionary of parameters for augmentation. Defaults to None.

        Returns:
            dict: A dictionary containing original and augmented images.
        """
        default_params = {
            "width": 640,
            "height": 640,
            "angle": 45,
            "blur_type": 'gaussian',
            "brightness": 0.3,
            "contrast": 0.3,
            "saturation": 0.3,
            "hue": 0.3,
            "noise_mean": 0,
            "noise_sigma": 25,
            "scale_fx": 1.2,
            "scale_fy": 1.2,
            "x_shift": 50,
            "y_shift": 50,
            "ksize": 5,
            "crop_size": (320, 320),
            "erase_area_ratio": 0.2,
            "cutmix_alpha": 1.0,
            "mixup_alpha": 1.0,
            "paste_size": (100, 100)
        }
        
        if params:
            default_params.update(params)
        
        results = {}
        original_img = self.img.copy()
        img2 = self.flip_horizontal(original_img)
        results['Original'] = original_img
        
        resized_img = self.resize_img(original_img, default_params["width"], default_params["height"])
        results['Resize'] = resized_img
        
        flipped_img = self.flip_horizontal(original_img)
        results['Flipped'] = flipped_img
        
        rotated_img = self.rotate_img(original_img, default_params["angle"])
        results['Rotated'] = rotated_img
        
        color_jittered_img = self.add_color_jitter(original_img,
            default_params["brightness"],
            default_params["contrast"],
            default_params["saturation"],
            default_params["hue"]
        )
        results['Color Jitter'] = color_jittered_img
        
        noisy_img = self.add_noise(original_img, default_params["noise_mean"], default_params["noise_sigma"])
        results['Noisy'] = noisy_img
        
        scaled_img = self.scale(original_img, default_params["scale_fx"], default_params["scale_fy"])
        results['Scaled'] = scaled_img
        
        translated_img = self.translate(original_img, default_params["x_shift"], default_params["y_shift"])
        results['Translated'] = translated_img
        
        blurred_img = self.apply_blur(original_img, default_params["blur_type"], default_params["ksize"])
        results['Blurred'] = blurred_img
        
        images_for_mosaic = [original_img] * 4
        results['Mosaic'] = self.mosaic(images_for_mosaic)

        cropped_img = self.random_crop(original_img, default_params["crop_size"])
        results['Random Crop'] = cropped_img

        erased_img = self.random_erasing(original_img, default_params["erase_area_ratio"])
        results['Random Erasing'] = erased_img

        cutmix_img = self.cutmix(original_img, img2, default_params["cutmix_alpha"])
        results['CutMix'] = cutmix_img

        mixup_img = self.mixup(original_img, img2, default_params["mixup_alpha"])
        results['MixUp'] = mixup_img

        pasted_img = self.copy_paste(original_img, img2, default_params["paste_size"])
        results["Pasted"] = pasted_img

        return results

