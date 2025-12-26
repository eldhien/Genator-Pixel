import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def load_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image_rgb(path, img_rgb):
    cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

def denoise(img_rgb):
    return cv2.medianBlur(img_rgb, 3)

def block_average(img_rgb, block_w=10, block_h=10):
    H, W = img_rgb.shape[:2]
    small_w = max(1, W // block_w)
    small_h = max(1, H // block_h)
    return cv2.resize(img_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)

def build_palette_from_small(small_img, max_colors=256):
    flat = small_img.reshape(-1, 3)
    dtype = np.dtype((np.void, flat.dtype.itemsize * 3))
    flat_view = np.ascontiguousarray(flat).view(dtype)
    uniq, inv, counts = np.unique(flat_view, return_inverse=True, return_counts=True)
    uniq_colors = uniq.view(flat.dtype).reshape(-1, 3)
    order = np.argsort(-counts)
    palette = uniq_colors[order][:max_colors]
    return palette.astype(np.uint8)

def knn_map_small_to_palette(small_img, palette):
    Hs, Ws = small_img.shape[:2]
    X_train = palette.astype(np.float32)
    y_train = np.arange(len(palette))
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    flat = small_img.reshape(-1, 3).astype(np.float32)
    preds = knn.predict(flat)
    return palette[preds].reshape(Hs, Ws, 3).astype(np.uint8)

def upscale_to_original(mapped_small, orig_shape):
    H, W = orig_shape[:2]
    return cv2.resize(mapped_small, (W, H), interpolation=cv2.INTER_NEAREST)

def add_outline(img_rgb, orig_rgb, strength=0.6):
    gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    out = img_rgb.astype(np.float32)
    out[edges > 0] *= (1.0 - strength)
    return np.clip(out, 0, 255).astype(np.uint8)
"""
def pipeline(input_path):
    orig = load_image(input_path)
    den = denoise(orig)
    small = block_average(den)
    palette = build_palette_from_small(small)
    mapped_small = knn_map_small_to_palette(small, palette)
    up = upscale_to_original(mapped_small, orig.shape)
    final = add_outline(up, den)
    return final
"""
def pipeline(input_path, block_size=10, palette_size=384, outline_strength=0.85):
    orig = load_image(input_path)
    den = denoise(orig)
    small = block_average(den, block_w=block_size, block_h=block_size)  # 👈 gunakan block_size
    palette = build_palette_from_small(small, max_colors=palette_size)   # 👈 gunakan palette_size
    mapped_small = knn_map_small_to_palette(small, palette)
    up = upscale_to_original(mapped_small, orig.shape)
    final = add_outline(up, den, strength=outline_strength)              # 👈 gunakan outline_strength
    return final

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python pixel_art.py <input> <output>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Gunakan parameter yang disesuaikan untuk gambar anime
    result = pipeline(input_path, block_size=10, palette_size=384, outline_strength=0.85)
    save_image_rgb(output_path, result)
"""
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python pixel_art.py <input> <output>")
        sys.exit(1)
    result = pipeline(sys.argv[1])
    save_image_rgb(sys.argv[2], result)
"""
