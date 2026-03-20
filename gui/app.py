import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms import lut, slicing, mapping
import deep_learning.infer as cnn_infer
import deep_learning.pretrained_infer as zhang_infer


# Constants
PREVIEW_SIZE  = 320

ALGORITHMS = [
    "LUT: Jet",
    "LUT: Plasma",
    "LUT: Viridis",
    "LUT: Hot",
    "LUT: Turbo",
    "LUT: Custom (B-G-R)",
    "Slicing: 4 levels",
    "Slicing: 6 levels",
    "Slicing: 8 levels",
    "Slicing: Background preserved",
    "Mapping: Sine",
    "Mapping: Density",
    "Mapping: Gamma + Plasma",
    "Mapping: Histogram EQ",
    "── CNN ──",
    "CNN: Auto colorization",
    "── Pretrained ──",
    "Pretrained: Zhang et al. 2016",
]

class PseudoColorStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("Pseudo-Color Studio")
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.root.geometry("1200x780")
        self.root.minsize(1000, 680)

        self.original_color = None
        self.original_gray  = None
        self.result_bgr     = None
        self.photo_original = None
        self.photo_result   = None

        self._build_ui()
        self._load_cnn()

    # ── UI Construction ────────────────────────────────────

    def _build_ui(self):
        # ── Top toolbar ──
        toolbar = ctk.CTkFrame(self.root, corner_radius=0, height=50)
        toolbar.pack(fill="x", padx=0, pady=0)

        ctk.CTkLabel(toolbar, text="Pseudo-Color Studio",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(side="left", padx=16, pady=10)

        ctk.CTkButton(toolbar, text="Open image", width=100, command=self._open_image).pack(side="left", padx=6)
        ctk.CTkButton(toolbar, text="Save result", width=100, command=self._save_result).pack(side="left", padx=6)
        ctk.CTkButton(toolbar, text="Save all results", width=120, command=self._save_all).pack(side="left", padx=6)

        self.status_var = ctk.StringVar(value="No image loaded")
        ctk.CTkLabel(toolbar, textvariable=self.status_var, font=ctk.CTkFont(size=12)).pack(side="right", padx=16)

        # ── Main area: left controls + right previews ──
        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=12, pady=12)

        self._build_controls(main)
        self._build_previews(main)

    def _build_controls(self, parent):
        ctrl = ctk.CTkFrame(parent, width=280)
        ctrl.pack(side="left", fill="y", padx=(0, 10), pady=0)
        ctrl.pack_propagate(False)

        ctk.CTkLabel(ctrl, text="Algorithm", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=14, pady=(14, 4))

        # Algorithm radio buttons in scrollable frame
        self.algo_var = ctk.StringVar(value=ALGORITHMS[0])
        
        scroll_frame = ctk.CTkScrollableFrame(ctrl, height=300)
        scroll_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        for algo in ALGORITHMS:
            if algo.startswith("──"):
                ctk.CTkLabel(scroll_frame, text=algo, text_color="gray", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", pady=(8, 2))
            else:
                rb = ctk.CTkRadioButton(scroll_frame, text=algo, variable=self.algo_var, value=algo, command=self._on_algo_change)
                rb.pack(anchor="w", pady=4)

        # ── Sliders section ──
        ctk.CTkLabel(ctrl, text="Parameters", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=14, pady=(10, 6))

        # Sine frequency
        self._slider_label(ctrl, "Sine frequency", "freq_display")
        self.freq_var = ctk.DoubleVar(value=1.0)
        self.freq_slider = ctk.CTkSlider(ctrl, variable=self.freq_var, from_=0.5, to=5.0, command=self._apply_slider)
        self.freq_slider.pack(fill="x", padx=14, pady=(0, 10))
        self.freq_var.trace_add("write", lambda *_: self._update_label(self.freq_display, f"{self.freq_var.get():.1f}"))

        # Gamma
        self._slider_label(ctrl, "Gamma", "gamma_display")
        self.gamma_var = ctk.DoubleVar(value=1.5)
        self.gamma_slider = ctk.CTkSlider(ctrl, variable=self.gamma_var, from_=0.3, to=4.0, command=self._apply_slider)
        self.gamma_slider.pack(fill="x", padx=14, pady=(0, 10))
        self.gamma_var.trace_add("write", lambda *_: self._update_label(self.gamma_display, f"{self.gamma_var.get():.1f}"))

        # Number of slices
        self._slider_label(ctrl, "Slice count", "slices_display")
        self.slices_var = ctk.IntVar(value=6)
        self.slices_slider = ctk.CTkSlider(ctrl, variable=self.slices_var, from_=2, to=8, number_of_steps=6, command=self._apply_slider)
        self.slices_slider.pack(fill="x", padx=14, pady=(0, 10))
        self.slices_var.trace_add("write", lambda *_: self._update_label(self.slices_display, str(self.slices_var.get())))

        spacer = ctk.CTkFrame(ctrl, height=2, fg_color="gray30")
        spacer.pack(fill="x", padx=14, pady=15)

        ctk.CTkButton(ctrl, text="Apply", command=self._apply).pack(fill="x", padx=14, pady=(0, 8))
        ctk.CTkButton(ctrl, text="Apply all & compare", fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), command=self._compare_all).pack(fill="x", padx=14, pady=(0, 14))

    def _build_previews(self, parent):
        prev = ctk.CTkFrame(parent, fg_color="transparent")
        prev.pack(side="left", fill="both", expand=True)

        # Labels row
        labels = ctk.CTkFrame(prev, fg_color="transparent")
        labels.pack(fill="x")
        
        ctk.CTkLabel(labels, text="Original Image", font=ctk.CTkFont(size=14)).pack(side="left", expand=True)
        ctk.CTkLabel(labels, text="Grayscale Input", font=ctk.CTkFont(size=14)).pack(side="left", expand=True)
        ctk.CTkLabel(labels, text="Colorized Output", font=ctk.CTkFont(size=14)).pack(side="left", expand=True)

        # Canvas row
        canvases = ctk.CTkFrame(prev, fg_color="transparent")
        canvases.pack(fill="both", expand=True, pady=6)

        self.frame_source = ctk.CTkFrame(canvases)
        self.frame_source.pack(side="left", expand=True, fill="both", padx=(0, 6))
        self.canvas_source = self._make_canvas(self.frame_source)

        self.frame_orig = ctk.CTkFrame(canvases)
        self.frame_orig.pack(side="left", expand=True, fill="both", padx=(0, 6))
        self.canvas_orig = self._make_canvas(self.frame_orig)

        self.frame_result = ctk.CTkFrame(canvases)
        self.frame_result.pack(side="left", expand=True, fill="both", padx=(0, 0))
        self.canvas_result = self._make_canvas(self.frame_result)

        # Info bar
        self.info_var = ctk.StringVar(value="Open an image to begin")
        ctk.CTkLabel(prev, textvariable=self.info_var, font=ctk.CTkFont(size=12)).pack(pady=4)

    # ── Helper Widgets ─────────────────────────────────────

    def _slider_label(self, parent, text, attr_name):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", padx=14, pady=(6, 0))
        ctk.CTkLabel(f, text=text, font=ctk.CTkFont(size=12)).pack(side="left")
        lbl = ctk.CTkLabel(f, text="", font=ctk.CTkFont(size=12))
        lbl.pack(side="right")
        setattr(self, attr_name, lbl)
        
    def _update_label(self, label, text):
        label.configure(text=text)

    def _make_canvas(self, parent):
        c = tk.Canvas(parent, bg="#2b2b2b", highlightthickness=0)
        c.pack(fill="both", expand=True, padx=2, pady=2)
        return c

    # ── Core Actions ───────────────────────────────────────

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Open grayscale image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"), ("All", "*.*")]
        )
        if not path:
            return

        img_color = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_color is None:
            messagebox.showerror("Error", f"Could not load image:\n{path}")
            return

        self.original_color = img_color
        self.original_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        self.status_var.set(f"{os.path.basename(path)}  |  {img_color.shape[1]}×{img_color.shape[0]} px")
        self.info_var.set(f"Loaded: {os.path.basename(path)}")
        self._show_image(self.canvas_source, self.original_color, is_gray=False)
        self._show_image(self.canvas_orig, self.original_gray, is_gray=True)
        self._apply()
        
    def _apply_slider(self, _=None):
        self._apply()

    def _apply(self):
        if self.original_gray is None:
            return

        algo = self.algo_var.get()
        if algo.startswith("──"):
            return
            
        result = self._run_algorithm(algo, self.original_gray)

        if result is not None:
            self.result_bgr = result
            self._show_image(self.canvas_result, result, is_gray=False)
            self.info_var.set(f"Applied: {algo}")

    def _on_algo_change(self):
        self._apply()

    def _run_algorithm(self, algo: str, gray: np.ndarray) -> np.ndarray:
        try:
            if   algo == "LUT: Jet":                      return lut.apply_lut(gray, "jet")
            elif algo == "LUT: Plasma":                   return lut.apply_lut(gray, "plasma")
            elif algo == "LUT: Viridis":                  return lut.apply_lut(gray, "viridis")
            elif algo == "LUT: Hot":                      return lut.apply_lut(gray, "hot")
            elif algo == "LUT: Turbo":                    return lut.apply_lut(gray, "turbo")
            elif algo == "LUT: Custom (B-G-R)":           return lut.apply_custom_lut(gray, lut.build_custom_lut())
            elif algo == "Slicing: 4 levels":             return slicing.apply_level_slicing(gray, 4)
            elif algo == "Slicing: 6 levels":             return slicing.apply_level_slicing(gray, 6)
            elif algo == "Slicing: 8 levels":             return slicing.apply_level_slicing(gray, 8)
            elif algo == "Slicing: Background preserved": return slicing.apply_level_slicing(gray, self.slices_var.get(), background=True)
            elif algo == "Mapping: Sine":                 return mapping.apply_sin_mapping(gray, self.freq_var.get())
            elif algo == "Mapping: Density":              return mapping.apply_density_mapping(gray)
            elif algo == "Mapping: Gamma + Plasma":       return mapping.apply_gamma_mapped(gray, self.gamma_var.get())
            elif algo == "Mapping: Histogram EQ":         return mapping.apply_histogram_equalized_lut(gray)
            elif algo == "── CNN ──":                     return None   # separator, do nothing
            elif algo == "CNN: Auto colorization":
                if not cnn_infer.is_loaded():
                    messagebox.showwarning(
                        "CNN not available",
                        "Place best_model.pth in data/checkpoints/ and restart the app."
                    )
                    return None
                self.status_var.set("Running CNN inference...")
                self.root.update_idletasks()
                
                result_container = [None]
                done_event = threading.Event()

                def run_inference():
                    result_container[0] = cnn_infer.colorize(gray)
                    done_event.set()

                thread = threading.Thread(target=run_inference, daemon=True)
                thread.start()

                # Poll every 100ms so UI stays responsive
                while not done_event.wait(timeout=0.1):
                    self.root.update_idletasks()

                result = result_container[0]
                info   = cnn_infer.get_info()
                iter_info = f"step {info['global_step']:,}" if info.get('global_step', '?') != "?" else f"epoch {info.get('epoch', '?')}"
                self.status_var.set(
                    f"CNN done — {iter_info}, "
                    f"val loss {info['val_loss']:.4f}"
                )
                return result
            elif algo == "── Pretrained ──":
                return None
            elif algo == "Pretrained: Zhang et al. 2016":
                if not zhang_infer.is_loaded():
                    messagebox.showwarning(
                        "Pretrained model not available",
                        "Download model files to data/pretrained/ — see README for links."
                    )
                    return None
                self.status_var.set("Running Zhang et al. colorization...")
                self.root.update_idletasks()
                result = zhang_infer.colorize(gray)
                self.status_var.set("Zhang et al. colorization done")
                return result
        except Exception as e:
            messagebox.showerror("Algorithm error", str(e))
        return None
    
    def _load_cnn(self):
        # Load our model
        success = cnn_infer.load_model()
        if success:
            info = cnn_infer.get_info()
            iter_info = f"step {info['global_step']:,}" if info.get('global_step', '?') != "?" else f"epoch {info.get('epoch', '?')}"
            self.status_var.set(
                f"CNN ready — {iter_info}, val loss {info['val_loss']:.4f}"
            )
        else:
            self.status_var.set("CNN model not found — place best_model.pth in data/checkpoints/")

        # Load Zhang et al. pretrained
        zhang_infer.load_model()

    def _compare_all(self):
        if self.original_gray is None:
            messagebox.showinfo("No image", "Open an image first.")
            return

        win = ctk.CTkToplevel(self.root)
        win.title("All algorithms — comparison")
        win.geometry("1100x800")

        scroll = ctk.CTkScrollableFrame(win)
        scroll.pack(fill="both", expand=True, padx=10, pady=10)

        cols = 5
        thumb = 200
        valid_idx = 0
        for algo in (["Original"] + ALGORITHMS):
            if algo.startswith("──"):
                continue
                
            r, c = divmod(valid_idx, cols)
            frame = ctk.CTkFrame(scroll)
            frame.grid(row=r, column=c, padx=6, pady=6)
            valid_idx += 1

            if algo == "Original":
                img_np = cv2.cvtColor(self.original_gray, cv2.COLOR_GRAY2RGB)
            else:
                bgr = self._run_algorithm(algo, self.original_gray)
                if bgr is None:
                    continue
                img_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            pil = Image.fromarray(img_np).resize((thumb, thumb), Image.LANCZOS)
            photo = ImageTk.PhotoImage(pil)

            canvas = tk.Canvas(frame, width=thumb, height=thumb,
                                bg="#2b2b2b", highlightthickness=0)
            canvas.create_image(0, 0, anchor="nw", image=photo)
            canvas.image = photo
            canvas.pack(padx=2, pady=2)

            ctk.CTkLabel(frame, text=algo, font=ctk.CTkFont(size=11), wraplength=thumb).pack(pady=4)

    def _save_result(self):
        if self.result_bgr is None:
            messagebox.showinfo("Nothing to save", "Apply an algorithm first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")]
        )
        if path:
            cv2.imwrite(path, self.result_bgr)
            self.status_var.set(f"Saved → {os.path.basename(path)}")

    def _save_all(self):
        if self.original_gray is None:
            messagebox.showinfo("No image", "Open an image first.")
            return
        folder = filedialog.askdirectory(title="Choose output folder")
        if not folder:
            return
        saved_count = 0
        for algo in ALGORITHMS:
            if algo.startswith("──"): continue
            result = self._run_algorithm(algo, self.original_gray)
            if result is not None:
                safe = algo.replace(":", "").replace(" ", "_").replace("/", "-")
                cv2.imwrite(os.path.join(folder, f"{safe}.png"), result)
                saved_count += 1
        self.status_var.set(f"All results saved → {folder}")
        messagebox.showinfo("Done", f"Saved {saved_count} images to:\n{folder}")

    # ── Image Display ──────────────────────────────────────

    def _show_image(self, canvas: tk.Canvas, img: np.ndarray, is_gray: bool):
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(),  PREVIEW_SIZE)
        ch = max(canvas.winfo_height(), PREVIEW_SIZE)

        if is_gray:
            pil = Image.fromarray(img).convert("RGB")
        else:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Fit image inside canvas preserving aspect ratio
        iw, ih = pil.size
        scale = min(cw / iw, ch / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        pil = pil.resize((nw, nh), Image.LANCZOS)

        photo = ImageTk.PhotoImage(pil)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, anchor="center", image=photo)
        canvas.image = photo  # keep reference


# ── Entry point ───────────────────────────────────────────

def launch():
    root = ctk.CTk()
    app = PseudoColorStudio(root)
    root.mainloop()

if __name__ == "__main__":
    launch()