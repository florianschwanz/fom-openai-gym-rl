import time

from lib.argument_extractor import ArgumentExtractor
from lib.environment_enum import Environment
from lib.visual_component import VisualComponent


class FreewayRewardShaper():
    """
    Pixel-based reward shaper for Atari game Pong
    """

    # Colors contained in the game
    BLACK = WHEEL_COLOR = (0, 0, 0)
    STREET_COLOR = BACKGROUND_COLOR = DARK_GREY = (142, 142, 142)
    STRIP_COLOR = LIGHT_GREY = (214, 214, 214)
    START_LANE_COLOR = GREY = (170, 170, 170)
    CHICKEN_COLOR = MEDIAN_STRIP_COLOR = YELLOW = (252, 252, 84)
    CAR_1_COLOR = LIGHT_LIME = (210, 210, 64)
    CAR_2_COLOR = LIGHT_GREEN = (135, 183, 84)
    CAR_3_COLOR = LIGHT_RED = (184, 50, 50)
    CAR_4_COLOR = BLUE = (84, 92, 214)
    CAR_5_COLOR = BROWN = (162, 98, 33)
    CAR_6_COLOR = DARK_BLUE = (24, 26, 167)
    CAR_7_COLOR = LIGHT_PINK = (228, 111, 111)
    CAR_8_COLOR = OLIVE = (105, 105, 15)
    CAR_9_COLOR = LIME = (180, 231, 117)
    CAR_10_COLOR = RED = (167, 26, 26)

    # Important positions
    BLACK_BOARDER_LEFT_X = 7
    BLACK_BOARDER_TOP_Y = 23
    BLACK_BOARDER_BOTTOM_Y = 194

    MEDIAN_STRIP_TOP_Y = 102
    MEDIAN_STRIP_BOTTOM_Y = 104
    PLAYER_1_CHICKEN_X_MIN = 44
    PLAYER_1_CHICKEN_X_MAX = 50
    PLAYER_2_CHICKEN_X_MIN = 108
    PLAYER_2_CHICKEN_X_MAX = 113

    PLAYER_1_CHICKEN_CENTER_X = 47
    CAR_1_CENTER_Y = 176
    CAR_2_CENTER_Y = 159
    CAR_3_CENTER_Y = 143
    CAR_4_CENTER_Y = 126
    CAR_5_CENTER_Y = 111
    CAR_6_CENTER_Y = 94
    CAR_7_CENTER_Y = 79
    CAR_8_CENTER_Y = 62
    CAR_9_CENTER_Y = 46
    CAR_10_CENTER_Y = 30

    # Environments this reward shaper makes sense to use with
    ENVIRONMENTS = [
        Environment.FREEWAY_V0,
        Environment.FREEWAY_NO_FRAMESKIP_V0
    ]

    def check_environment(func):
        def check_environment_and_call(self, *args, **kwargs):
            """Checks if reward shaping is done on a matching environment"""
            environment = ArgumentExtractor.extract_argument(kwargs, "environment", None)

            if environment not in self.ENVIRONMENTS:
                raise Exception("Reward shaping method does match environment "
                                "(method:" + func.__name__ + ", environment:" + environment.value + ")")

            # Remove arguments that were only used for this wrapper
            kwargs.pop("environment", None)

            return func(self, *args, **kwargs)

        return check_environment_and_call

    def initialize_reward_shaper(func):
        def initialize_reward_shaper_and_call(self, *args, **kwargs):
            self.screen = ArgumentExtractor.extract_argument(kwargs, "screen", None)
            self.reward = ArgumentExtractor.extract_argument(kwargs, "reward", None)
            self.done = ArgumentExtractor.extract_argument(kwargs, "done", None)
            self.info = ArgumentExtractor.extract_argument(kwargs, "info", None)

            self.player_chicken_pixels, \
            self.car_1_pixels, \
            self.car_2_pixels, \
            self.car_3_pixels, \
            self.car_4_pixels, \
            self.car_5_pixels, \
            self.car_6_pixels, \
            self.car_7_pixels, \
            self.car_8_pixels, \
            self.car_9_pixels, \
            self.car_10_pixels = self.extract_pixels_optimized(self.screen)

            self.player_chicken = VisualComponent(self.player_chicken_pixels, self.screen)
            self.car_1 = VisualComponent(self.car_1_pixels, self.screen)
            self.car_2 = VisualComponent(self.car_2_pixels, self.screen)
            self.car_3 = VisualComponent(self.car_3_pixels, self.screen)
            self.car_4 = VisualComponent(self.car_4_pixels, self.screen)
            self.car_5 = VisualComponent(self.car_5_pixels, self.screen)
            self.car_6 = VisualComponent(self.car_6_pixels, self.screen)
            self.car_7 = VisualComponent(self.car_7_pixels, self.screen)
            self.car_8 = VisualComponent(self.car_8_pixels, self.screen)
            self.car_9 = VisualComponent(self.car_9_pixels, self.screen)
            self.car_10 = VisualComponent(self.car_10_pixels, self.screen)
            self.lives = self.info["ale.lives"]

            kwargs.pop("current_episode_reward", None)
            kwargs.pop("max_episode_reward", None)
            kwargs.pop("min_episode_reward", None)

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
            if (self.chicken.visible):
                print("DEBUG"
                      + " chicken " + str(self.chicken.center))
            else:
                print("DEBUG "
                      + " no chicken")
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
    def reward_chicken_vertical_position(self, **kwargs):
        """
        Gives an additional reward if the player's racket hits the ball
        :return: shaped reward
        """

        additional_reward = ArgumentExtractor.extract_argument(kwargs, "additional_reward", 0)

        reward_max = additional_reward
        reward_min = 0

        chicken_y_min = 0  # TODO Use the y coordinate when the player scores
        chicken_y_max = 210

        if self.player_chicken.visible:
            chicken_y = self.player_chicken.center[1]

            additional_reward = round(
                ((reward_max - reward_min) / (chicken_y_min - chicken_y_max) * chicken_y + reward_max), 4)
            return additional_reward
        else:
            return 0

    def extract_pixels(self, screen):
        """
        Extracts pixels from a screen
        :return: a dictionary having coordinates as key, and rgb values as value
        """

        player_chicken_pixels = []
        car_1_pixels = []
        car_2_pixels = []
        car_3_pixels = []
        car_4_pixels = []
        car_5_pixels = []
        car_6_pixels = []
        car_7_pixels = []
        car_8_pixels = []
        car_9_pixels = []
        car_10_pixels = []

        screen_height = screen.shape[0]  # y-axis starting from top-left corner
        screen_width = screen.shape[1]  # x-axis starting from top-left corner

        # Define relevant section of the screen
        section_x_min = self.BLACK_BOARDER_LEFT_X + 1
        section_x_max = screen_width
        section_y_min = self.BLACK_BOARDER_TOP_Y + 1
        section_y_max = self.BLACK_BOARDER_BOTTOM_Y - 1

        # Define step size
        steps_x = 2
        steps_y = 2

        for x in range(section_x_min, section_x_max, steps_x):
            for y in range(section_y_min, section_y_max, steps_y):
                coordinates = (x, y)
                value = (screen[y][x][0], screen[y][x][1], screen[y][x][2])

                if FreewayRewardShaper.is_background_pixel(self, x, y, value):
                    pass
                elif FreewayRewardShaper.is_player_chicken_pixel(self, x, y, value):
                    player_chicken_pixels.append(coordinates)
                elif FreewayRewardShaper.is_car_pixel(self, x, y, value, self.CAR_1_COLOR):
                    car_1_pixels.append(coordinates)
                elif FreewayRewardShaper.is_car_pixel(self, x, y, value, self.CAR_2_COLOR):
                    car_2_pixels.append(coordinates)
                elif FreewayRewardShaper.is_car_pixel(self, x, y, value, self.CAR_3_COLOR):
                    car_3_pixels.append(coordinates)
                elif FreewayRewardShaper.is_car_pixel(self, x, y, value, self.CAR_4_COLOR):
                    car_4_pixels.append(coordinates)
                elif FreewayRewardShaper.is_car_pixel(self, x, y, value, self.CAR_5_COLOR):
                    car_5_pixels.append(coordinates)
                elif FreewayRewardShaper.is_car_pixel(self, x, y, value, self.CAR_6_COLOR):
                    car_6_pixels.append(coordinates)
                elif FreewayRewardShaper.is_car_pixel(self, x, y, value, self.CAR_7_COLOR):
                    car_7_pixels.append(coordinates)
                elif FreewayRewardShaper.is_car_pixel(self, x, y, value, self.CAR_8_COLOR):
                    car_8_pixels.append(coordinates)
                elif FreewayRewardShaper.is_car_pixel(self, x, y, value, self.CAR_9_COLOR):
                    car_9_pixels.append(coordinates)
                elif FreewayRewardShaper.is_car_pixel(self, x, y, value, self.CAR_10_COLOR):
                    car_10_pixels.append(coordinates)

        return player_chicken_pixels, car_1_pixels, car_2_pixels, car_3_pixels, car_4_pixels, car_5_pixels, \
               car_6_pixels, car_7_pixels, car_8_pixels, car_9_pixels, car_10_pixels

    def extract_pixels_optimized(self, screen):
        """
        Extracts pixels from a screen
        :return: a dictionary having coordinates as key, and rgb values as value
        """

        player_chicken_pixels = []
        car_1_pixels = []
        car_2_pixels = []
        car_3_pixels = []
        car_4_pixels = []
        car_5_pixels = []
        car_6_pixels = []
        car_7_pixels = []
        car_8_pixels = []
        car_9_pixels = []
        car_10_pixels = []

        screen_height = screen.shape[0]  # y-axis starting from top-left corner
        screen_width = screen.shape[1]  # x-axis starting from top-left corner

        # Define relevant section of the screen
        section_x_min = self.BLACK_BOARDER_LEFT_X + 1
        section_x_max = screen_width
        section_y_min = self.BLACK_BOARDER_TOP_Y + 1
        section_y_max = self.BLACK_BOARDER_BOTTOM_Y - 1

        # Define step size
        steps_x = 1
        steps_y = 1

        for y in range(section_y_min, section_y_max, steps_y):
            x = self.PLAYER_1_CHICKEN_CENTER_X

            coordinates = (x, y)
            value = (screen[y][x][0], screen[y][x][1], screen[y][x][2])

            if FreewayRewardShaper.is_background_pixel(self, x, y, value):
                pass
            elif FreewayRewardShaper.is_player_chicken_pixel(self, x, y, value):
                player_chicken_pixels.append(coordinates)

        for x in range(section_x_min, section_x_max, steps_x):
            FreewayRewardShaper.append_car_pixel(self, screen, car_1_pixels, x, self.CAR_1_CENTER_Y, self.CAR_1_COLOR)
            FreewayRewardShaper.append_car_pixel(self, screen, car_2_pixels, x, self.CAR_2_CENTER_Y, self.CAR_2_COLOR)
            FreewayRewardShaper.append_car_pixel(self, screen, car_3_pixels, x, self.CAR_3_CENTER_Y, self.CAR_3_COLOR)
            FreewayRewardShaper.append_car_pixel(self, screen, car_4_pixels, x, self.CAR_4_CENTER_Y, self.CAR_4_COLOR)
            FreewayRewardShaper.append_car_pixel(self, screen, car_5_pixels, x, self.CAR_5_CENTER_Y, self.CAR_5_COLOR)
            FreewayRewardShaper.append_car_pixel(self, screen, car_6_pixels, x, self.CAR_6_CENTER_Y, self.CAR_6_COLOR)
            FreewayRewardShaper.append_car_pixel(self, screen, car_7_pixels, x, self.CAR_7_CENTER_Y, self.CAR_7_COLOR)
            FreewayRewardShaper.append_car_pixel(self, screen, car_8_pixels, x, self.CAR_8_CENTER_Y, self.CAR_8_COLOR)
            FreewayRewardShaper.append_car_pixel(self, screen, car_9_pixels, x, self.CAR_9_CENTER_Y, self.CAR_9_COLOR)
            FreewayRewardShaper.append_car_pixel(self, screen, car_10_pixels, x, self.CAR_10_CENTER_Y,
                                                 self.CAR_10_COLOR)

        return player_chicken_pixels, car_1_pixels, car_2_pixels, car_3_pixels, car_4_pixels, car_5_pixels, \
               car_6_pixels, car_7_pixels, car_8_pixels, car_9_pixels, car_10_pixels

    def append_car_pixel(self, screen, car_pixels, x, y, car_color):
        coordinates = (x, y)
        value = (screen[y][x][0], screen[y][x][1], screen[y][x][2])
        if FreewayRewardShaper.is_car_pixel(self, x, y, value, car_color):
            car_pixels.append(coordinates)

    def is_background_pixel(self, x, y, value):
        return value == self.BACKGROUND_COLOR

    def is_player_chicken_pixel(self, x, y, value):
        return value == self.CHICKEN_COLOR \
               and x >= self.PLAYER_1_CHICKEN_X_MIN and x <= self.PLAYER_1_CHICKEN_X_MAX \
               and y != self.MEDIAN_STRIP_TOP_Y \
               and y != self.MEDIAN_STRIP_BOTTOM_Y

    def is_car_pixel(self, x, y, value, car_color):
        return value == car_color
