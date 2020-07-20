import random
import colorsys
import numpy as np

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def overlay_mask(image, mask, color=(0, 255, 0), alpha=1):
    """Overlay 2d mask on rgb image with color and transparency

    Args:
        image:
        mask:
        color:
        alpha:

    Returns:

    """
    assert len(mask.shape) == 2
    assert len(image.shape) == 3
    assert mask.shape == image.shape[:2]
    canvas_opqaue = np.where(np.expand_dims(mask, axis=2),
                      np.array(color).astype(np.uint8).reshape([1, 1, 3]), image)
    canvas = (canvas_opqaue * alpha + image * (1 - alpha)).astype(np.uint8)
    return canvas