import copy

import gym
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from utils import convert_to_power_of_2, place_sprite

NO_ROAD = 0
HORIZONTAL = 1
VERTICAL = 2


class Track(object):
    def __init__(self, width=256, height=256, game_id=0):
        # everything is easier if we use powers of 2
        width = convert_to_power_of_2(width)
        height = convert_to_power_of_2(height)

        self.game_id = game_id

        # height and width of the entire observation
        self.width = width
        self.height = height

        # this is the high level design of the track
        # (0's for bg, 1's for horizontal track, 2's for vertical track)
        self.track_bitmap = self.create_track_bitmap()
        self.num_road_tiles = np.sum(self.track_bitmap != NO_ROAD)

        # the image for the background (solid green for now)
        self.bg_image = self.get_background_image(self.width, self.height)

        # compute the "scale_factor" between the track_bitmap and the full size image
        self.width_pixels_per_bit, self.height_pixels_per_bit = self.compute_pixels_per_bit()

        # get the image for a road tile
        # make it so one bit in the track bitmap equals one road tile
        self.road_tile_width, self.road_tile_height = self.width_pixels_per_bit, self.height_pixels_per_bit
        self.road_tile_im = self.get_road_tile_image(self.road_tile_width, self.road_tile_height)

        # make full track by laying the road tiles in the pattern specified by the bitmap
        self.image, self.full_bitmap, self.reward_bitmap, self.done_bitmap = self.lay_track(self.bg_image,
                                                                                            self.track_bitmap,
                                                                                            self.road_tile_im)

    def get_all_valid_locations_for_sprite(self, sprite_size):
        """Loop through entire track and store all valid coordinates that
         the sprite can be in aka it has to stay fully on the road"""

        valid_coords = []
        full_bitmap_height, full_bitmap_width = self.full_bitmap.shape
        for y_coord in range(full_bitmap_height):
            for x_coord in range(full_bitmap_width):
                if self.is_valid_sprite_location(sprite_size, (x_coord, y_coord)):
                    valid_coords.append((x_coord, y_coord))
        return np.asarray(valid_coords)

    def is_valid_sprite_location(self, sprite_size, location):
        """Check if placing sprite in location will be valid
            aka will be fully on the road"""

        loc_x, loc_y = location

        if loc_x < 0 or loc_y < 0 or loc_x > self.image.size[0] or loc_y > self.image.size[1]:
            return False

        region_of_occupation = self.get_sprite_region_of_occupation(sprite_size, location)

        is_valid = np.all(self.full_bitmap[region_of_occupation] != NO_ROAD)
        return is_valid

    def get_sprite_region_of_occupation(self, sprite_size, location):
        """get the bitmap values for every portion of map that sprite is occupying"""

        sprite_width, sprite_height = sprite_size
        loc_x, loc_y = location
        # remember numpy arrays are indexed by [y,x] and not [x,y]
        region_of_occupation = np.s_[loc_y: loc_y + sprite_height, loc_x: loc_x + sprite_width]
        return region_of_occupation

    def is_sprite_location_vertical(self, sprite_size, location):
        """check if any portion of the space the sprite is occupying is on a vertical road tile"""

        region_of_occupation = self.get_sprite_region_of_occupation(sprite_size, location)
        is_vertical = np.any(self.full_bitmap[region_of_occupation] == VERTICAL)
        return is_vertical

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
        horiz_track = np.concatenate((np.zeros(1, ), HORIZONTAL * np.ones(width - 2), np.zeros(1, )))
        right_vert = np.concatenate((np.zeros(width - 2), VERTICAL * np.ones(1, ), np.zeros(1, )))
        left_vert = np.concatenate((np.zeros(1, ), VERTICAL * np.ones(1, ), np.zeros(width - 2)))

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
        bg_image = bg_image.resize((width, height))
        return bg_image

    def get_road_tile_image(self, width, height):
        """Load PIL image from disk for a track piece"""
        road_tile_im = Image.open("images/road_tile.png")
        road_tile_im = road_tile_im.resize((width, height))
        road_tile_im = road_tile_im.rotate(90)  # rotate to make horizontal
        return road_tile_im

    def lay_track(self, bg_im, track_bitmap, road_tile_im):
        """Lays all the track pieces onto an image in the shape specified by the bitmao

        Arguments:
            bg_im (PIL Image): the background image
            track_bitmap (np.ndarray): array specifying pattern of track and grass
            road_tile_im (PIL Image): one primitive track piece
                                    should be size (width_scale_factor, height_scale_factor)

        Basically we loop over the track_bitmap and place road_tiles in places the bitmap specifies

        Returns:

        """
        track_im = bg_im.copy()

        # remember ordering of axes for numpy arrays vs. PIL images!
        bitmap_height, bitmap_width = track_bitmap.shape
        tile_w, tile_h = road_tile_im.size

        tile_count = 0
        # create full size bitmap -> one bit per pixel
        full_bitmap = np.zeros_like(np.asarray(track_im)[:, :, 0])
        reward_bitmap = np.zeros_like(np.asarray(track_im)[:, :, 0])
        done_bitmap = np.zeros_like(np.asarray(track_im)[:, :, 0])
        for bit_x in range(bitmap_height):
            for bit_y in range(bitmap_width):
                tile_type = track_bitmap[bit_y, bit_x]

                # if a road tile belongs in this location
                if tile_type in [HORIZONTAL, VERTICAL]:
                    tile_x, tile_y = self.get_tile_coordinates(bit_x, bit_y)

                    # the index for all pixels that this tile occupies
                    tile_region = np.s_[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w]

                    # flip the bit for every pixel of the full_bitmap where the tile lays
                    full_bitmap[tile_region] = tile_type

                    # place tile image on background image
                    self.place_tile_on_track(track_im,
                                             road_tile_im,
                                             (tile_x, tile_y),
                                             tile_type)

                    # designate reward if needed in this location
                    reward_bitmap = self.place_reward(reward_bitmap, tile_count, tile_region)
                    done_bitmap = self.update_done_bitmap(done_bitmap, tile_count, tile_region)

                    tile_count += 1

        return track_im, full_bitmap, reward_bitmap, done_bitmap

    def update_done_bitmap(self, done_bitmap, tile_count, tile_region):
        """Add reward to the reward bitmap"""
        tile_geographic_index = self.get_tile_geographic_index(tile_count)
        y_region, x_region = tile_region
        if tile_geographic_index == "upper_left":
            # the whole left part of the upper left tile
            # every y location and the left most x location
            terminal_region = np.s_[y_region, x_region.start]
            done_bitmap[terminal_region] = 1

        elif tile_geographic_index == "bottom_right":
            # the whole right part of the bottom right tile
            # every y location and the rightmost x location
            terminal_region = np.s_[y_region, x_region.stop - 1]
            done_bitmap[terminal_region] = 1

        # else reward is 0

        return done_bitmap

    def place_reward(self, reward_bitmap, tile_count, tile_region):
        """Add reward to the reward bitmap"""
        tile_geographic_index = self.get_tile_geographic_index(tile_count)
        y_region, x_region = tile_region
        if tile_geographic_index == "upper_left":
            reward = (4 if self.game_id == 0 else 0)

            # the whole left part of the upper left tile
            # every y location and the left most x location
            reward_region = np.s_[y_region, x_region.start]
            reward_bitmap[reward_region] = reward

        elif tile_geographic_index == "bottom_right":
            reward = 2

            # the whole right part of the bottom right tile
            # every y location and the rightmost x location
            reward_region = np.s_[y_region, x_region.stop - 1]
            reward_bitmap[reward_region] = reward

        # else reward is 0

        return reward_bitmap

    def get_tile_geographic_index(self, tile_count):
        """Get the general location (upper left, bottom right, somewhere in the middle)
         of the tile based on how many tiles have been placed"""

        if tile_count == 0:
            tile_loc = "upper_left"
        elif tile_count == self.num_road_tiles - 1:
            tile_loc = "bottom_right"
        else:
            tile_loc = "middle_somewhere"

        return tile_loc

    def get_tile_coordinates(self, bit_x, bit_y):
        """we place one tile per bit of the bitmap, so each time we increment
         the coordinates of the bitmap, we increment the pixel index
         of the background by the dimension length of the road tile"""

        tile_x, tile_y = (int(self.road_tile_width * bit_x),
                          int(self.road_tile_height * bit_y))

        return tile_x, tile_y

    def place_tile_on_track(self, full_track_im, tile_im, tile_location, tile_type):
        """Paste the road tile image on the track image"""

        # place horizontal tile
        if tile_type == HORIZONTAL:
            full_track_im.paste(tile_im, tile_location)
        # place vertical tile (rotate horizontal tile)
        elif tile_type == VERTICAL:
            full_track_im.paste(tile_im.rotate(90), tile_location)


class Car(object):
    def __init__(self, track, width=32, height=32, step_size=1):
        self.width = width
        self.height = height
        self.size = (self.width, self.height)
        self.step_size = step_size
        self.image = Image.open("images/car.png").resize(self.size)
        self.track = copy.deepcopy(track)
        self.valid_locations = track.get_all_valid_locations_for_sprite(self.size)
        self.location = (0, 0)

    def reset(self, location):
        self.location = copy.deepcopy(location)
        self.image = Image.open("images/car.png").resize(self.size)

    def _move(self, location, direction, step_size):
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

        return new_x, new_y

    def get_as_close_as_you_can(self, direction):
        """"inch forward with size 1 steps until you reach the end of valid track"""

        new_location = self.location
        trial_location = self.location
        steps = 0
        while self.track.is_valid_sprite_location(self.size, trial_location) and steps <= self.step_size:
            new_location = copy.deepcopy(trial_location)
            trial_location = self._move(new_location, direction, step_size=1)
            steps += 1
        return new_location

    def move(self, direction):
        new_location = self._move(self.location, direction, self.step_size)

        # if new location is not a valid location on the track
        # get as close to new location as possible
        if not self.track.is_valid_sprite_location(self.size, new_location):
            new_location = self.get_as_close_as_you_can(direction)

        # rotate sprite if transitioning from horizontal to vertical or vice versa
        before_is_vertical = self.track.is_sprite_location_vertical(self.size, self.location)
        after_is_vertical = self.track.is_sprite_location_vertical(self.size, new_location)
        if before_is_vertical != after_is_vertical:
            self.image = self.image.rotate(90)

        self.location = new_location


class CarNav(gym.Env):
    def __init__(self, width=256, height=256, step_size=10, reset_location="random", game_id=0):
        self.width = width
        self.height = height
        self.game_id = game_id
        self.reset_location = reset_location

        self.track = Track(width, height, game_id=game_id)

        # make car half the size of each road tile which is the size observation size / 8
        car_width, car_height = self.track.road_tile_width // 2, self.track.road_tile_height // 2
        self.car = Car(self.track, width=car_width, height=car_height, step_size=step_size)

        self.action_space = gym.spaces.Discrete(4)
        self.action_meanings = ["up", "down", "left", "right"]

        self.done = False

        self.valid_track_positions = self.track.get_all_valid_locations_for_sprite(self.car.size)
        self.reward_bitmap = self.track.reward_bitmap

    def _get_obs(self):
        obs = place_sprite(base_image=self.track.image,
                           sprite_image=self.car.image,
                           sprite_location=self.car.location)
        return np.asarray(obs)

    def _get_reward(self):
        car_region = self.track.get_sprite_region_of_occupation(self.car.size, self.car.location)
        rewards_at_this_region = self.reward_bitmap[car_region]

        # any pixels the car occupies are a nonzero reward, we get that reward
        reward = np.max(rewards_at_this_region)
        return reward

    def _is_done(self):
        car_region = self.track.get_sprite_region_of_occupation(self.car.size, self.car.location)
        is_done = np.any(self.track.done_bitmap[car_region])
        return is_done

    def _get_start_location(self):
        num_valid_locations = self.valid_track_positions.shape[0]
        if self.reset_location == "random":
            location_ind = np.random.choice(num_valid_locations)
        elif self.reset_location == "constant":
            location_ind = num_valid_locations // 2
        location = tuple(self.valid_track_positions[location_ind])
        return location

    def reset(self):
        self.done = False
        start_location = self._get_start_location()
        self.car.reset(start_location)
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


if __name__ == "__main__":
    UP, DOWN, LEFT, RIGHT = range(4)

    actions = [RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, UP, UP, LEFT, LEFT, LEFT, LEFT, LEFT, LEFT, LEFT]

    env = CarNav(step_size=32)

    obs = env.reset()
    plt.imshow(obs)
    plt.axis("off")
    plt.show()
    reward = 0
    for action in actions:
        obs, reward, done, _ = env.step(action)
        print("Action taken was {}".format(env.action_meanings[action]))
        print("Reward is: {}".format(reward))
        print("Done is: {}".format(done))
        plt.imshow(obs)
        plt.axis("off")
        plt.show()
