import cv2
import numpy as np
import random

class YOLODataAugmentation:
    def __init__(self, img_path: str = "") -> None:
        self.img = cv2.imread(img_path)
        if self.img is None:
            raise ValueError(f"Image at path '{img_path}' cannot be loaded. Check the path and try again.")

    def resize_img(self, img: str, width: float, height: float):
        return cv2.resize(img, (width, height))

    def flip_horizontal(self, img: str):
        return cv2.flip(img, 1)

    def rotate_img(self, img: str, angle: float):
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        return cv2.warpAffine(img, rotation_matrix, (w, h))

    def add_color_jitter(self, img: str, brightness: float=0.3, contrast: float=0.3, saturation: float=0.3, hue: float=0.3):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 2] *= 1 + random.uniform(-brightness, brightness)
        hsv[..., 1] *= 1 + random.uniform(-saturation, saturation)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def add_noise(self, img: str, mean: float=0, sigma: float=25):
        noise = np.random.normal(mean, sigma, img.shape).astype(np.uint8)
        return cv2.add(img, noise)

    def mosaic(self, images: str, size: tuple=(640, 640)):
        assert len(images) == 4, "Mosaic requires exactly 4 images."
        h, w = size[0] // 2, size[1] // 2
        images_resized = [cv2.resize(img, (w, h)) for img in images]
        top = np.hstack(images_resized[:2])
        bottom = np.hstack(images_resized[2:])
        return np.vstack((top, bottom))

    def scale(self, img: str, fx: float=1.2, fy: float=1.2):
        return cv2.resize(img, None, fx=fx, fy=fy)

    def translate(self, img: str, x_shift: float=50, y_shift: float=50):
        (h, w) = img.shape[:2]
        translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        return cv2.warpAffine(img, translation_matrix, (w, h))

    def apply_blur(self, img: str, blur_type: str='gaussian', ksize: int=5):
        if blur_type == 'gaussian':
            return cv2.GaussianBlur(img, (ksize, ksize), 0)
        elif blur_type == 'median':
            return cv2.medianBlur(img, ksize)
        elif blur_type == 'bilateral':
            return cv2.bilateralFilter(img, ksize, 75, 75)

    def apply_all(self, params=None):
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
            "ksize": 5
        }
        
        if params:
            default_params.update(params)
        
        results = {}
        original_img = self.img.copy()
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

        return results