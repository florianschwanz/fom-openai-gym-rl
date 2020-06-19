import time

from environment_enum import Environment
from visual_component import VisualComponent


class SpaceInvadersRewardShaper():
    """
    Pixel-based reward shaper for Atari game Space Invaders
    """

    # Colors contained in the game
    BLACK = BACKGROUND_COLOR = (0, 0, 0)
    GREEN = SPACESHIP_COLOR = (50, 132, 50)
    GOLD = POINTS_RIGHT_COLOR = (162, 134, 56)
    OLIVE = ALIENS_COLOR = (134, 134, 29)
    GREY = RAY_COLOR = (142, 142, 142)
    ORANGE = ROCKS_COLOR = (181, 83, 40)
    DARKGREEN = FLOOR_COLOR = (80, 89, 22)

    # Important positions
    SCORE_Y_MIN = 10
    SCORE_Y_MAX = 19
    FLOOR_Y_MIN = 195
    FLOOR_Y_MAX = 219
    LINE_IN_FLOOR_Y_MIN = 196
    LINE_IN_FLOOR_Y_MAX = 200
    ROCK_ONE_X_MIN = 42
    ROCK_ONE_X_MAX = 49
    ROCK_TWO_X_MIN = 74
    ROCK_TWO_X_MAX = 81
    ROCK_THREE_X_MIN = 106
    ROCK_THREE_X_MAX = 113

    ALIENS_AREA_Y_MAX = 31

    # Environments this reward shaper makes sense to use with
    ENVIRONMENTS = [
        Environment.SPACE_INVADERS_V0,
        Environment.SPACE_INVADERS_NO_FRAMESKIP_V0
    ]

    def check_environment(func):
        def check_environment_and_call(self, *args, **kwargs):
            """Checks if reward shaping is done on a matching environment"""
            environment = kwargs["environment"]

            if environment not in self.ENVIRONMENTS:
                raise Exception("Reward shaping method does match environment "
                                "(method:" + func.__name__ + ", environment:" + environment.value + ")")

            # Remove arguments that were only used for this wrapper
            kwargs.pop("environment", None)

            return func(self, *args, **kwargs)

        return check_environment_and_call

    def initialize_reward_shaper(func):
        def initialize_reward_shaper_and_call(self, *args, **kwargs):
            self.screen = kwargs["screen"]
            self.reward = kwargs["reward"]
            self.done = kwargs["done"]
            self.info = kwargs["info"]

            self.spaceship_pixels, \
            self.rocks_pixels, \
            self.rays_pixels, \
            aliens_pixels = SpaceInvadersRewardShaper.extract_pixels(self, self.screen)

            self.spaceship = VisualComponent(self.spaceship_pixels, self.screen)
            self.rays = VisualComponent(self.rays_pixels, self.screen)
            self.lives = self.info["ale.lives"]

            # Remove arguments that were only used for this wrapper
            kwargs.pop("screen", None)
            kwargs.pop("reward", None)
            kwargs.pop("done", None)
            kwargs.pop("info", None)

            return func(self, *args, **kwargs)

        return initialize_reward_shaper_and_call

    def debug_positions(func):
        def debug_positions_and_call(self, *args, **kwargs):
            """
            Prints positions of all elements on the screen
            :param self:
            :param args:
            :param kwargs:
            :return:
            """
            print("DEBUG player spaceship" + str(self.spaceship.center))

            return func(self, *args, **kwargs)

        return debug_positions_and_call

    def debug_time(func):
        def debug_time_and_call(self, *args, **kwargs):
            start_time = time.time()
            ret = func(self, *args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            print("DEBUG " + str(round(duration, 4)) + " / " + str(func.__name__))
            return ret

        return debug_time_and_call

    @check_environment
    @initialize_reward_shaper
    def reward_player_avoids_line_of_fire(self, additional_reward=0.025):
        """
        Gives an additional reward if the player's spaceship avoids line of fire
        :return: shaped reward
        """

        if self.spaceship.visible and self.rays.visible:
            spaceship_x_values = self.get_x_values(self.spaceship_pixels)
            rocks_x_values = self.get_x_values(self.rocks_pixels)
            rays_x_values = self.get_x_values(self.rays_pixels)

            spaceship_in_line_with_rays = any(x in spaceship_x_values for x in rays_x_values)
            spaceship_in_line_with_rocks = any(x in spaceship_x_values for x in rocks_x_values)
            if not spaceship_in_line_with_rays or spaceship_in_line_with_rocks:
                return additional_reward
            else:
                return 0
        else:
            return 0

    def extract_pixels(self, screen):
        """
        Extracts pixels from an screen
        :return: a dictionary having coordinates as key, and rgb values as value
        """

        spaceship_pixels = []
        rocks_pixels = []
        rays_pixels = []
        aliens_pixels = []

        screen_height = screen.shape[0]  # y-axis starting from top-left corner
        screen_width = screen.shape[1]  # x-axis starting from top-left corner

        # Define relevant section of the screen
        section_x_min = 0
        section_x_max = screen_width
        section_y_min = self.ALIENS_AREA_Y_MAX
        section_y_max = self.FLOOR_Y_MIN

        # Define step size
        steps_x = 1
        steps_y = 3

        for x in range(section_x_min, section_x_max, steps_x):
            for y in range(section_y_min, section_y_max, steps_y):
                coordinates = (x, y)
                value = (screen[y][x][0], screen[y][x][1], screen[y][x][2])

                if SpaceInvadersRewardShaper.is_backgrund_pixel(self, x, y, value):
                    pass
                elif SpaceInvadersRewardShaper.is_spaceship_pixel(self, x, y, value):
                    spaceship_pixels.append(coordinates)
                elif SpaceInvadersRewardShaper.is_rocks_pixel(self, x, y, value):
                    rocks_pixels.append(coordinates)
                elif SpaceInvadersRewardShaper.is_rays_pixel(self, x, y, value):
                    rays_pixels.append(coordinates)
                elif SpaceInvadersRewardShaper.is_aliens_pixel(self, x, y, value):
                    aliens_pixels.append(coordinates)

        return spaceship_pixels, rocks_pixels, rays_pixels, aliens_pixels

    def is_backgrund_pixel(self, x, y, value):
        return value == self.BACKGROUND_COLOR

    def is_spaceship_pixel(self, x, y, value):
        return value == self.SPACESHIP_COLOR \
               and (y > self.SCORE_Y_MAX and y < self.LINE_IN_FLOOR_Y_MIN)

    def is_rocks_pixel(self, x, y, value):
        return value == self.ROCKS_COLOR

    def is_rays_pixel(self, x, y, value):
        return value == self.RAY_COLOR \
               and (y > self.SCORE_Y_MAX and y < self.LINE_IN_FLOOR_Y_MIN)

    def is_aliens_pixel(self, x, y, value):
        return value == self.ALIENS_COLOR

    def get_x_values(self, pixels):
        """
        Extracts x values from map of pixels
        :param pixels pixels to extract x values from
        :return: list of x values
        """
        x_values = []

        for pixel in pixels:
            if pixel[0] not in x_values:
                x_values.append(pixel[0])
        return x_values
