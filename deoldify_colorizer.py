import os
from ..config import MODEL_ROOT_DEOLDIFY
from deoldify.visualize import get_image_colorizer
from PIL import Image
import torch
from pathlib import Path

# Patch torch.load to handle DeOldify checkpoint loading
original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False  # Ensures full model loading
    return original_torch_load(*args, **kwargs)


torch.load = patched_torch_load

# Initialize DeOldify colorizer (artistic mode)
colorizer = get_image_colorizer(
    artistic=True, root_folder=Path(MODEL_ROOT_DEOLDIFY)  # points to DeOldify
)


def colorize_bw_image(input_path, output_path):
    """
    Colorizes a black and white image using DeOldify.

    Args:
        input_path (str): Path to the input grayscale image.
        output_path (str): Path to save the colorized output.
    """
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    result_image = colorizer.get_transformed_image(
        path=input_path, render_factor=35, post_process=True
    )

    # Save colorized result
    result_image.save(output_path)
