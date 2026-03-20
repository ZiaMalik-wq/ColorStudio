import torch
import numpy as np
import cv2
from skimage import color
from deep_learning.model import ColorizationUNet
import config
import os


class ColorizationInference:
    """
    Loads the trained U-Net and runs colorization inference
    on a single grayscale image. CPU-friendly for local use.
    """

    def __init__(self):
        self.device     = torch.device("cpu")
        self.model      = None
        self.AB_NORM    = config.AB_NORM
        self.IMAGE_SIZE = config.IMAGE_SIZE
        self.loaded     = False
        self.model_info = {}

    def load_model(self, checkpoint_path: str = None) -> bool:
        """
        Load model weights from checkpoint.
        Returns True if successful, False otherwise.
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR, "best_model.pth"
            )

        if not os.path.exists(checkpoint_path):
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Handle both wrapped checkpoints and plain state_dict files.
            state_dict = checkpoint.get("model_state", checkpoint)
            state_dict = {
                k.replace("module.", ""): v
                for k, v in state_dict.items()
            }

            # Build the model to match the checkpoint architecture.
            bottleneck_weight = state_dict.get("bottleneck.net.0.weight")
            bottleneck_ch = (
                int(bottleneck_weight.shape[0])
                if bottleneck_weight is not None
                else 1024
            )
            self.model = ColorizationUNet(bottleneck_ch=bottleneck_ch)
            self.model.load_state_dict(state_dict)
            self.model.eval()

            self.loaded     = True

            # Default val_loss to float to avoid formatting errors downstream.
            val_loss = checkpoint.get("val_loss", 0.0)
            if not isinstance(val_loss, float):
                try:
                    val_loss = float(val_loss)
                except (TypeError, ValueError):
                    val_loss = 0.0

            self.model_info = {
                "epoch":       checkpoint.get("epoch",    "?"),
                "global_step": checkpoint.get("global_step", "?"),
                "val_loss":    val_loss,
                "path":        checkpoint_path,
            }
            return True

        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def colorize(self, gray_img: np.ndarray) -> np.ndarray:
        """
        Colorize a grayscale image.
        Input  : HxW or HxWxC uint8 numpy array
        Output : HxWx3 BGR uint8 numpy array (original resolution)
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        original_h, original_w = gray_img.shape[:2]

        # Preprocess once and reuse.
        gray = self._prepare_gray(gray_img)

        resized     = cv2.resize(gray, (self.IMAGE_SIZE, self.IMAGE_SIZE),
                                 interpolation=cv2.INTER_LINEAR)
        rgb_for_lab = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        lab         = color.rgb2lab(rgb_for_lab)
        L           = (lab[:, :, 0] / 50.0) - 1.0
        L_tensor    = torch.tensor(L, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Inference
        with torch.no_grad():
            AB_tensor = self.model(L_tensor)

        AB_np = AB_tensor.squeeze().permute(1, 2, 0).numpy() * self.AB_NORM


        AB_np = np.clip(AB_np, -110, 110)
        
        rgb_full  = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        lab_full  = color.rgb2lab(rgb_full)
        L_full    = np.clip(lab_full[:, :, 0], 0, 100)

        # Upsample AB to the original resolution.
        AB_upsampled = cv2.resize(
            AB_np,
            (original_w, original_h),
            interpolation=cv2.INTER_CUBIC
        )

        lab_final = np.concatenate([L_full[:, :, None], AB_upsampled], axis=2)
        rgb_final = np.clip(color.lab2rgb(lab_final.astype(np.float64)), 0, 1)
        bgr_final = cv2.cvtColor((rgb_final * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        return bgr_final

    def is_loaded(self) -> bool:
        return self.loaded

    def get_info(self) -> dict:
        return self.model_info

    def _prepare_gray(self, img: np.ndarray) -> np.ndarray:
        """
        Ensure image is a 2D uint8 grayscale array.
        Handles: HxW, HxWx1, HxWx3 inputs, float or uint8.
        """
        # Flatten to 2D if needed
        if img.ndim == 3:
            img = img[:, :, 0]

        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        return img


    # Module-level singleton shared across the app.
_inferencer = ColorizationInference()


def load_model(checkpoint_path: str = None) -> bool:
    return _inferencer.load_model(checkpoint_path)


def colorize(gray_img: np.ndarray) -> np.ndarray:
    return _inferencer.colorize(gray_img)


def is_loaded() -> bool:
    return _inferencer.is_loaded()


def get_info() -> dict:
    return _inferencer.get_info()