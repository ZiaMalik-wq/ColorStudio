import cv2
import numpy as np
import os


class ZhangColorizer:
    """
    Wrapper around the Zhang et al. (2016) pretrained colorization model.
    Loaded via OpenCV DNN — no PyTorch needed.
    Uses the same LAB color space as our U-Net for fair comparison.
    """

    def __init__(self):
        self.net    = None
        self.pts    = None
        self.loaded = False

    def _is_valid_prototxt(self, prototxt_path: str) -> bool:
        # Guard against common bad downloads that contain HTTP error text.
        try:
            if os.path.getsize(prototxt_path) < 200:
                return False
            with open(prototxt_path, "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(512)
            if "404: Not Found" in head or "<html" in head.lower():
                return False
            return "layer" in head or "name:" in head
        except OSError:
            return False

    def load_model(self, model_dir: str = None) -> bool:
        if model_dir is None:
            base     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base, "data", "pretrained")

        prototxt   = os.path.join(model_dir, "colorization_deploy_v2.prototxt")
        caffemodel = os.path.join(model_dir, "colorization_release_v2.caffemodel")
        pts_file   = os.path.join(model_dir, "pts_in_hull.npy")

        if not all(os.path.exists(f) for f in [prototxt, caffemodel, pts_file]):
            print("Pretrained model files not found in:", model_dir)
            return False

        if not self._is_valid_prototxt(prototxt):
            print("Invalid prototxt file:", prototxt)
            print("Re-download from:")
            print("https://raw.githubusercontent.com/richzhang/colorization/caffe/models/colorization_deploy_v2.prototxt")
            return False

        try:
            self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

            # Load cluster centers for AB quantization
            pts = np.load(pts_file).transpose().reshape(2, 313, 1, 1)

            # Add cluster centers as convolution weights
            self.net.getLayer(self.net.getLayerId("class8_ab")).blobs = [
                pts.astype(np.float32)
            ]
            self.net.getLayer(self.net.getLayerId("conv8_313_rh")).blobs = [
                np.full([1, 313], 2.606, dtype=np.float32)
            ]

            self.loaded = True
            return True

        except Exception as e:
            print(f"Failed to load pretrained model: {e}")
            return False

    def colorize(self, gray_img: np.ndarray) -> np.ndarray:
        """
        Colorize a grayscale image using Zhang et al. pretrained model.

        Args:
            gray_img: Grayscale image (H, W) or (H, W, 1), uint8

        Returns:
            BGR image (H, W, 3) uint8
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        original_h, original_w = gray_img.shape[:2]

        # Prepare grayscale
        if gray_img.ndim == 3:
            gray_img = gray_img[:, :, 0]

        # Convert to LAB — model expects L channel normalized to [0, 1]
        gray_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        gray_lab = cv2.cvtColor(gray_rgb.astype(np.float32) / 255.0,
                                cv2.COLOR_RGB2Lab)
        L        = gray_lab[:, :, 0]

        # Resize to 224x224 (model's required input size)
        L_resized = cv2.resize(L, (224, 224))
        L_resized -= 50   # mean centering

        # Run inference
        blob = cv2.dnn.blobFromImage(L_resized)
        self.net.setInput(blob)
        AB_dec = self.net.forward()[0, :, :, :].transpose((1, 2, 0))

        # Resize AB back to original size
        AB_us = cv2.resize(AB_dec, (original_w, original_h))

        # Combine original L with predicted AB
        result_lab        = np.zeros((original_h, original_w, 3), dtype=np.float32)
        result_lab[:,:,0] = L
        result_lab[:,:,1] = AB_us[:,:,0]
        result_lab[:,:,2] = AB_us[:,:,1]

        # Convert back to BGR
        result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_Lab2BGR)
        result_bgr = np.clip(result_bgr * 255, 0, 255).astype(np.uint8)

        return result_bgr

    def is_loaded(self) -> bool:
        return self.loaded


# Module-level singleton
_zhang = ZhangColorizer()

def load_model(model_dir: str = None) -> bool:
    return _zhang.load_model(model_dir)

def colorize(gray_img: np.ndarray) -> np.ndarray:
    return _zhang.colorize(gray_img)

def is_loaded() -> bool:
    return _zhang.is_loaded()