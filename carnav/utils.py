from pathlib import Path

import numpy as np
from PIL import Image





def place_sprite(base_image, sprite_image, sprite_location):
    """ places sprite at certain location on bg

    Arguments:
        base_image (PIL Image): background image
        sprite_image (PIL Image): sprite image (can be RGB or RGBA)
        sprite_location (tuple): upper left x,y coords to place sprite

    Returns:
        final_im (PIL Image): final image with sprite pasted on it
    """
    final_im = base_image.copy()
    mask = sprite_image if sprite_image.mode == "RGBA" else None
    final_im.paste(sprite_image, sprite_location, mask=mask)
    return final_im


def convert_to_power_of_2(num):
    log_2_of_num = np.log2(num)
    is_true_power_of_2 = log_2_of_num == np.floor(log_2_of_num)
    if is_true_power_of_2:
        return num
    else:
        exponent_to_use = np.round(log_2_of_num)
        return int(2 ** exponent_to_use)

def make_transparent(im_path):
    """assumes background is white and makes it transparent

    Arguments:
        im_path (str): string with path to image with white bg
    """

    # get save dir by grabbing the parent directory of the image
    save_dir = str(Path(im_path).parent)
    im_name = str(Path(im_path).name)

    # load as PIL im then convert to numpy array
    pil_im = Image.open(im_path)
    np_im = np.asarray(pil_im)
    transparent_np_im = add_transparent_bg_to_numpy_im(np_im)

    # convert to PIL im
    transparent_pil_im = Image.fromarray(transparent_np_im)

    # save in same directory as original image
    transparent_im_name = "_".join(["transparent", im_name])
    transparent_im_path = Path(save_dir) / Path(transparent_im_name)
    transparent_pil_im.save(transparent_im_path)


def add_transparent_bg_to_numpy_im(np_im):
    """takes RGB numpy array and converts white bg to be transparent

    Arguments:
        np_im (np.ndarray; dtype=uint8):
            RGB or RGBA numpy array, which has a white background
            Shape should be (height,width,3) or (height,width,4)

    Returns:
        rgba_array (np.ndarray; dtype=uint8):
            RGBA numpy array, with transparent background
            Shape should be (height,width,4)
            and the last channel should be 0 at locations where white bg was

    """
    # first three channels is RGB. if there is a 4th it is alpha, which specifies transparency
    rgb_part = np.copy(np_im[:, :, :3])

    # make our own alpha channel (255 is most opaque, 0 is most transparent)
    alpha = 255 * np.ones_like(rgb_part[:, :, 0])

    # 255 for R,G,and B is white, so we check which pixels are [255,255,255], which means white
    white_mask = np.all(rgb_part == 255, axis=2)

    # set the transparency to 100% (alpha = 0) for any white pixels
    alpha[white_mask] = 0

    # concat rgb with alpha to create RGBA array
    alpha = np.expand_dims(alpha, axis=2)  # newshape: (h,w,1)
    rgba_array = np.concatenate((rgb_part, alpha), axis=2)
    return rgba_array


if __name__ == "__main__":
    import os

    print(os.getcwd())
    make_transparent("./images/car.png")
