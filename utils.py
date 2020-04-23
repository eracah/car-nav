from PIL import Image
from pathlib import Path
import numpy as np



def convert_to_power_of_2(num):
    log_2_of_num = np.log2(num)
    is_true_power_of_2 = log_2_of_num == np.floor(log_2_of_num)
    if is_true_power_of_2:
        return num
    else:
        exponent_to_use = np.round(log_2_of_num)
        return int(2**exponent_to_use)

def create_reward_channel(track_bitmap, game_id=0):
    """Create bit map for rewards (map each grid square to a reward).

    Arguments:
    ----------

    track_bitmap : array-like
                    bitmap of the track
    game_id : int
              id of game if 0 then top left reward is 10, if the id is 1 then it is 0

    """

    height = track_bitmap.shape[0]
    reward_channel = np.zeros_like(track_bitmap)

    # the x,y coordinate of the first (top left) moveable grid square (aka the top left part of the track) and the bottom right one
    top_left_y, top_left_x = track_ind_to_coords(0, track_bitmap)
    bottom_right_y, bottom_right_x = track_ind_to_coords(-1, track_bitmap)

    # make top left part of track have reward 10 and bottom right have reward of 1
    reward_channel[top_left_y, top_left_x] = 10 if game_id == 0 else 0
    reward_channel[bottom_right_y, bottom_right_x] = 1

    return reward_channel




def create_agent_channel(track_bitmap):
    """Create agent position bitmap (all zeros except for where agent is)

    Arguments:
    ----------

    track_bitmap : array-like
                    bitmap of the track

    """

    # the agent will be set to be in the middle of the track
    num_track_positions = len(track_bitmap[track_bitmap == 0])
    agent_start_ind = num_track_positions // 2  # np.random.choice(num_track_positions)
    agent_start_y, agent_start_x = track_ind_to_coords(agent_start_ind, track_bitmap)

    # set all grid squares to zero except the one where the agent is
    agent_channel = np.zeros_like(track_bitmap)
    agent_channel[agent_start_y, agent_start_x] = 1
    return agent_channel

def track_ind_to_coords(ind, track_bitmap):
    """Return the x,y coordinates of the track bitmap given an index.
    The track has a finite number of naviagble grid squares, so if we count them starting from the top left,
    then we can enumerate each valid grid square with an index
    This function converts that index to the corresponding x,y coordinate in the track bitmap

    Arguments:
    ----------

    ind : int
          the index of the grid square on the track (counting from top left and moving right and down)

    track_bitmap : 2D array-like
                  the bitmap (2D array) corresponding to the track
    """

    valid_track_value = 0
    height, width = track_bitmap.shape

    # create arrays of all possible coordinates in the track
    x_coord, y_coord = np.meshgrid(np.arange(width),
                                   np.arange(height))
    # of the coordina
    y, x = y_coord[track_bitmap == valid_track_value][ind], \
           x_coord[track_bitmap == valid_track_value][ind]
    return y, x



def place_sprite(bg, sprite, location):
    """ places sprite at certain location on bg

    Arguments:
        bg (PIL Image): background image
        sprite (PIL Image): sprite image (can be RGB or RGBA)

    Returns:
        final_im (PIL Image): final image with sprite pasted on it
    """
    final_im = bg.copy()
    mask = sprite if sprite.mode == "RGBA" else None
    final_im.paste(sprite, location, mask=mask)
    return final_im




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
    #first three channels is RGB. if there is a 4th it is alpha, which specifies transparency
    rgb_part = np.copy(np_im[:, :, :3])

    # make our own alpha channel (255 is most opaque, 0 is most transparent)
    alpha = 255*np.ones_like(rgb_part[:,:,0])

    # 255 for R,G,and B is white, so we check which pixels are [255,255,255], which means white
    white_mask = np.all(rgb_part==255, axis=2)

    # set the transparency to 100% (alpha = 0) for any white pixels
    alpha[white_mask] = 0

    # concat rgb with alpha to create RGBA array
    alpha = np.expand_dims(alpha,axis=2) # newshape: (h,w,1)
    rgba_array = np.concatenate((rgb_part,alpha), axis=2)
    return rgba_array

if __name__ == "__main__":
    import os
    print(os.getcwd())
    make_transparent("./images/car.png")




