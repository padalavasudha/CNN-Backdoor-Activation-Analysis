"""
Trigger injection for building poisoned training data.

The trigger used in this project is a solid red square placed in the
bottom-right corner of the image, sized relative to the image's
smaller dimension. This is a classic, well-studied patch-trigger style
(similar in spirit to the original BadNets trigger) chosen for its
simplicity and visual clarity in demos.
"""
from PIL import Image, ImageDraw


def add_trigger(img_path: str, save_path: str, size_ratio: float = 0.2) -> None:
    """Stamp a red-square trigger onto an image and save the result.

    Args:
        img_path: path to the source image.
        save_path: path to write the triggered image to.
        size_ratio: trigger edge length as a fraction of min(width, height).
            Clamped to a minimum of 4 pixels so tiny images still get a
            visible trigger.
    """
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    size = max(4, int(min(w, h) * size_ratio))
    draw.rectangle([w - size, h - size, w - 1, h - 1], fill=(255, 0, 0))
    img.save(save_path)
