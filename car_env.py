import gym
import numpy as np
import copy
from PIL import Image
import scipy
import copy
import random

from utils import convert_to_power_of_2, place_sprite


class Track(object):
    def __init__(self, width=256, height=256):
        
        # everything is easier if we use powers of 2
        width = convert_to_power_of_2(width)
        height = convert_to_power_of_2(height)
        
        # height and width of the entire observation
        self.width = width
        self.height = height

        # this is the high level design of the track
        # (0's for bg, 1's for horizontal track, 2's for vertical track)
        self.track_bitmap = self.create_track_bitmap()

        # the image for the background (solid green for now)
        self.bg_image = self.get_background_image(self.width, self.height)

        # compute the "scale_factor" between the track_bitmap and the full size image
        self.width_pixels_per_bit, self.height_pixels_per_bit = self.compute_pixels_per_bit()

        # get the image for a road tile
        # make it so one bit in the track bitmap equals one road tile
        self.road_tile_width, self.road_tile_height = self.width_pixels_per_bit, self.height_pixels_per_bit
        self.road_tile_im = self.get_road_tile_image(self.road_tile_width, self.road_tile_height)
        
        # make full track by laying the road tiles in the pattern specified by the bitmap
        self.image, self.full_bitmap = self.lay_track(self.bg_image, self.track_bitmap, self.road_tile_im)


    def get_all_valid_locations(self, sprite_size):
        """Loop through entire track and store all valid coordinates that
         the sprite can be in aka it has to stay fully on the road"""

        valid_coords = []
        full_bitmap_height, full_bitmap_width = self.full_bitmap.shape
        for y_coord in range(full_bitmap_height):
            for x_coord in range(full_bitmap_width):
                if self.is_valid_location(sprite_size,(x_coord,y_coord)):
                    valid_coords.append((x_coord, y_coord))
        return np.asarray(valid_coords)


    def is_valid_location(self, sprite_size, location):
        """Check if placing sprite in location will be valid
            aka will be fully on the track"""


        sprite_width, sprite_height = sprite_size
        loc_x, loc_y = location

        if loc_x < 0 or loc_y < 0:
            return False
        if loc_x > self.image.size[0] or loc_y > self.image.size[1]:
            return False

        # remember numpy arrays are indexed by [y,x] and not [x,y]
        region_of_occupation = self.full_bitmap[loc_y:loc_y + sprite_height, loc_x:loc_x + sprite_width, ]
        is_valid = np.all(region_of_occupation)
        return is_valid

    def compute_pixels_per_bit(self):
        """Computes how many pixels each bit of the track bit map represent"""

        # PIL images are ordered width, then height
        bg_width, bg_height = self.bg_image.size

        # note numpy arrays are ordered height first then width
        bitmap_height, bitmap_width = self.track_bitmap.shape

        # compute the pixels per bit in the bit map
        height_pixels_per_bit = bg_height // bitmap_height
        width_pixels_per_bit = bg_width // bitmap_width

        return width_pixels_per_bit, height_pixels_per_bit




    def create_track_bitmap(self, width=8):
        """Create bit map for the car track.
         Zeros are the background and are non-navigable
         Ones are horizontal pieces
         Twos are vertical pieces

        Arguments:
        ----------

        width : int
                the width of observation space aka the number of columns in the track array (default = 8)
        """

        border = np.zeros(width)
        horiz_track = np.concatenate((np.zeros(1, ), np.ones(width - 2), np.zeros(1, )))
        right_vert = np.concatenate((np.zeros(width - 2), 2*np.ones(1, ), np.zeros(1, )))
        left_vert = np.concatenate((np.zeros(1, ), 2*np.ones(1, ), np.zeros(width - 2)))

        track_bitmap = np.stack((border,
                                  horiz_track,
                                  right_vert,
                                  horiz_track,
                                  left_vert,
                                  left_vert,
                                  horiz_track,
                                  border)
                                 ).astype(np.int8)
        return track_bitmap


    def get_background_image(self, width, height):
        """Load PIL image from disk for the background"""

        bg_image = Image.open("images/background.png")
        bg_image = bg_image.resize( (width, height))
        return bg_image

    def get_road_tile_image(self, width, height):
        """Load PIL image from disk for a track piece"""
        road_tile_im = Image.open("images/road_tile.png")
        road_tile_im = road_tile_im.resize((width, height))
        road_tile_im = road_tile_im.rotate(90) # rotate to make horizontal
        return road_tile_im


    def lay_track(self, bg_im, track_bitmap, road_tile_im):
        """Lays all the track pieces onto an image in the shape specified by the bitmao

        Arguments:
            bg_im (PIL Image): the background image
            track_bitmap (np.ndarray): array specifying pattern of track and grass
            road_tile_im (PIL Image): one primitive track piece
                                    should be size (width_scale_factor, height_scale_factor)

        Basically we loop over the track_bitmap and place road_tiles in places the bitmap specifies
        """
        full_track = bg_im.copy()

        # remember ordering of axes for numpy arrays vs. PIL images!
        bitmap_height, bitmap_width = track_bitmap.shape
        road_tile_width, road_tile_height = road_tile_im.size

        # create full size bitmap -> one bit per pixel
        full_bitmap = np.zeros_like(np.asarray(full_track)[:,:,0])
        for bit_x in range(bitmap_height):
            for bit_y in range(bitmap_width):

                # we place one tile per bit of the bitmap, so each time we increment
                # the coordinates of the bitmap, we increment the pixel index
                # of the background by the dimension length of the road tile
                bg_x_coord, bg_y_coord = int(road_tile_width * bit_x), int(road_tile_height * bit_y)

                tile_location = (bg_x_coord, bg_y_coord)
                # place horizontal tile
                if track_bitmap[bit_y, bit_x] == 1:
                    full_track.paste(road_tile_im, tile_location)
                    full_bitmap[bg_y_coord : bg_y_coord + road_tile_height,
                                    bg_x_coord:bg_x_coord + road_tile_width] = 1

                # place vertical tile (rotate horizontal tile)
                elif track_bitmap[bit_y, bit_x] == 2:
                    full_track.paste(road_tile_im.rotate(90), tile_location)
                    full_bitmap[bg_y_coord: bg_y_coord + road_tile_height,
                    bg_x_coord:bg_x_coord + road_tile_width] = 2

        return full_track, full_bitmap


class Car(object):
    def __init__(self, track, width=32, height=32, step_size=1):
        self.width = width
        self.height = height
        self.size = (self.width, self.height)
        self.step_size = step_size
        self.image = Image.open("images/car.png").resize(self.size)
        self.track = copy.deepcopy(track)
        self.valid_locations = track.get_all_valid_locations(self.size)
        self.location = (0, 0)
        self.reset()

    def reset(self):
        num_valid_locations = self.valid_locations.shape[0]
        location_ind = num_valid_locations // 2 #np.random.choice(num_valid_locations)
        self.location = tuple(self.valid_locations[location_ind])


    def _move(self,location, direction, step_size):
        cur_x, cur_y = location
        new_x, new_y = copy.deepcopy(cur_x), copy.deepcopy(cur_y)
        if direction == "up":
            new_y -= step_size
        elif direction == "down":
            new_y += step_size
        elif direction == "left":
            new_x -= step_size
        elif direction == "right":
            new_x += step_size
        else:
            assert False, "{} is an invalid direction".format(direction)

        return (new_x, new_y)


    def get_as_close_as_you_can(self, direction):
        """"inch forward with size 1 steps until you reach the end of valid track"""
        new_location = self.location
        trial_location = self.location
        steps = 0
        while self.track.is_valid_location(self.size, trial_location) and steps <= self.step_size:
            new_location = copy.deepcopy(trial_location)
            trial_location = self._move(new_location, direction, step_size=1)
            steps += 1
        return new_location


    def move(self, direction):
        new_location = self._move(self.location, direction, self.step_size)

        # if new location is not a valid location on the track
        # get as close to new location as possible
        if not self.track.is_valid_location(self.size, new_location):
            new_location = self.get_as_close_as_you_can(direction)

        self.location = new_location




class CarNav(gym.Env):
    def __init__(self, width=256, height=256, step_size=10, game_id=0):
        self.width = width
        self.height = height
        self.game_id = game_id

        self.track = Track(width, height)

        # make car same size as each road tile which is the size observation size / 8
        car_width, car_height = self.track.road_tile_width, self.track.road_tile_height
        self.car = Car(self.track, width=car_width, height=car_height, step_size=step_size)


        self.action_space = gym.spaces.Discrete(4)
        self.action_meanings = ["up", "down", "left", "right"]

        self.done = False

        self.valid_track_positions = self.track.get_all_valid_locations(self.car.size)
        self.top_left_pos, self.bottom_right_pos = tuple(self.valid_track_positions[0]),\
                                                   tuple(self.valid_track_positions[-1])



    def _get_obs(self):
        obs = place_sprite(self.track.image, self.car.image, self.car.location)
        return np.asarray(obs)

    def _get_reward(self):
        if self.car.location == self.top_left_pos:
            if self.game_id == 0:
                reward = 4
            else:
                reward = 0
        elif self.car.location == self.bottom_right_pos:
            reward = 2
        else:
            reward = 0
        return reward


    def _is_done(self):
        if self.car.location == self.top_left_pos or self.car.location == self.bottom_right_pos:
            done = True
        else:
            done = False
        return done


    def reset(self):
        self.done = False
        self.car.reset()
        obs = self._get_obs()
        return obs


    def step(self, action):
        if self.done:
            return self._get_obs(), 0, self.done, {}
        action_str = self.action_meanings[action]
        self.car.move(action_str)
        obs = self._get_obs()
        reward = self._get_reward()
        self.done = self._is_done()

        return obs, reward, self.done, {}
