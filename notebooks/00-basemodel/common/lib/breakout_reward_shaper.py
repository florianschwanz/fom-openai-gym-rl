import math
import statistics

from environment_enum import Environment


class Component:
    """
    Represents a distinct visual entity on the screen
    """

    def __init__(self, pixels, screen):
        if len(pixels) > 0:
            self.visible = True
            self.center = BreakoutRewardShaper.get_component_center(pixels)
            self.left = BreakoutRewardShaper.get_component_left(screen, pixels)
            self.right = BreakoutRewardShaper.get_component_right(pixels)
        else:
            self.visible = False
            self.center = ()
            self.left = ()
            self.right = ()


class BreakoutRewardShaper():
    """
    Pixel-based reward shaper for Atari game Breakout
    """

    # Colors contained in the game
    BLACK = (0, 0, 0)
    GREY = (142,142,142)
    RED = (200,72,72)
    ORANGE = (198,108,58)
    YELLOW = (180, 122,48)
    LIME = (162,162,42)
    GREEN = (72,160,72)
    BLUE = (66,72,200)
    TEAL = (66,158,130)

    # Important positions
    LINE_RED_Y_MIN = 57
    LINE_RED_Y_MAX = 62
    PLAYER_RACKET_Y_MIN = 189
    PLAYER_RACKET_Y_MAX = 194

    BALL_CENTER_Y_WHEN_PLAYED_BY_PLAYER = 184

    # Environments this reward shaper makes sense to use with
    ENVIRONMENTS = [
        Environment.BREAKOUT_V0,
    ]

    def check_environment(func):
        def check_environment_and_call(self, *args, **kwargs):
            """Checks if reward shaping is done on a matching environment"""
            environment_name = kwargs["environment_name"]

            if environment_name not in self.ENVIRONMENTS:
                raise Exception("Reward shaping method does match environment "
                                "(method:" + func.__name__ + ", environment:" + environment_name.value + ")")

            # Remove arguments that were only used for this wrapper
            kwargs.pop("environment_name", None)

            return func(self, *args, **kwargs)

        return check_environment_and_call

    def initialize_reward_shaper(func):
        def initialize_reward_shaper_and_call(self, *args, **kwargs):
            self.screen = kwargs["screen"]
            self.reward = kwargs["reward"]
            self.done = kwargs["done"]
            self.info = kwargs["info"]

            self.pixels = BreakoutRewardShaper.extract_pixels(self.screen)
            self.colors = BreakoutRewardShaper.extract_colors(self.pixels)

            self.ball = Component(BreakoutRewardShaper.get_ball_pixels(self, self.pixels), self.screen)
            self.player_racket = Component(BreakoutRewardShaper.get_player_racket_pixels(self, self.pixels), self.screen)
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
            if (self.ball.visible):
                print("DEBUG"
                      + " player " + str(self.player_racket.center)
                      + " ball " + str(self.ball.center))
            else:
                print("DEBUG "
                      + " player " + str(self.player_racket.center)
                      + " no ball")
            return func(self, *args, **kwargs)

        return debug_positions_and_call

    @check_environment
    @initialize_reward_shaper
    def reward_player_racket_hits_ball(self, additional_reward=0.025):
        """
        Gives an additional reward if the player's racket hits the ball
        :return: shaped reward
        """

        if self.ball.visible and self.player_racket.visible \
                and self.ball.center[1] == self.BALL_CENTER_Y_WHEN_PLAYED_BY_PLAYER \
                and self.player_racket.left[0] <= self.ball.center[0] <= self.player_racket.right[0]:
            return additional_reward
        else:
            return 0

    @check_environment
    @initialize_reward_shaper
    def reward_player_racket_covers_ball(self, additional_reward=0.025):
        """
        Gives an additional reward if the player's racket covers y-coordinate of the ball
        :return: shaped reward
        """

        if self.ball.visible and self.player_racket.visible \
                and self.player_racket.left[0] <= self.ball.center[0] <= self.player_racket.right[0]:
            return additional_reward
        else:
            return 0

    @check_environment
    @initialize_reward_shaper
    def reward_player_racket_close_to_ball_linear(self, additional_reward=0.05):
        reward_max = additional_reward
        reward_min = 0

        dist_max = 160
        dist_min = 0

        if self.ball.visible and self.player_racket.visible:
            dist = abs(self.ball.center[0] - self.player_racket.center[0])
            additional_reward = round(((reward_max - reward_min) / (dist_min - dist_max) * dist + reward_max), 4)
            return additional_reward
        else:
            return 0

    @check_environment
    @initialize_reward_shaper
    def reward_player_racket_close_to_ball_quadratic(self, additional_reward=0.05):
        reward_max = math.sqrt(additional_reward)
        reward_min = 0

        dist_max = 160
        dist_min = 0

        if self.ball.visible and self.player_racket.visible:
            dist = abs(self.ball.center[0] - self.player_racket.center[0])
            additional_reward = round(((reward_max - reward_min) / (dist_min - dist_max) * dist + reward_max), 4)
            return math.pow(additional_reward, 2)
        else:
            return 0

    def extract_pixels(screen):
        """
        Extracts pixels from an screen
        :return: a dictionary having coordinates as key, and rgb values as value
        """
        pixels = {}

        screen_height = screen.shape[0]  # y-axis starting from top-left corner
        screen_width = screen.shape[1]  # x-axis starting from top-left corner
        # screen_dim = screen.shape[2]

        for h in range(screen_height):
            for w in range(screen_width):
                coordinates = (w, h)  # Flip with and height here to match regular x-y syntax
                value = (screen[h][w][0], screen[h][w][1], screen[h][w][2])

                pixels[coordinates] = value
        return pixels

    def extract_colors(pixels):
        """
        Extracts distinct colors from map of pixels
        :param pixels pixels to extract colors from
        :return: list of distinct colors
        """
        colors = []

        for color in pixels.values():
            c = str(color[0]) + "," + str(color[1]) + "," + str(color[2])
            if c not in colors:
                colors.append(c)
        return colors

    def get_ball_pixels(self, pixels):
        """
        Gets all pixels that represent the ball by color
        :return: list of pixels representing the ball
        """

        BALL_COLOR = BreakoutRewardShaper.RED

        ball_pixels = []

        for key, value in pixels.items():
            x = key[0]
            y = key[1]

            if value == BALL_COLOR \
                    and not (y >= self.LINE_RED_Y_MIN and y <= self.LINE_RED_Y_MAX) \
                    and not (y >= self.PLAYER_RACKET_Y_MIN and y <= self.PLAYER_RACKET_Y_MAX):
                ball_pixels.append(key)

        return ball_pixels

    def get_player_racket_pixels(self, pixels):
        """
        Gets all pixels that represent the player's racket by color
        :return: list of pixels representing the the player's racket
        """

        RACKET_COLOR = BreakoutRewardShaper.RED

        racket_pixels = []

        for key, value in pixels.items():
            x = key[0]
            y = key[1]

            if value == RACKET_COLOR \
                    and not (y >= self.LINE_RED_Y_MIN and y <= self.LINE_RED_Y_MAX) \
                    and (y >= self.PLAYER_RACKET_Y_MIN and y <= self.PLAYER_RACKET_Y_MAX):
                racket_pixels.append(key)

        return racket_pixels

    def get_opponent_racket_pixels(pixels):
        """
        Gets all pixels that represent the opponent's racket by color
        :return: list of pixels representing the the opponent's racket
        """

        RACKET_COLOR = BreakoutRewardShaper.ORANGE

        racket_pixels = []

        for key, value in pixels.items():
            x = key[0]
            y = key[1]

            if value == RACKET_COLOR \
                    and (y >= 21):
                racket_pixels.append(key)

        return racket_pixels

    def get_component_center(pixels):
        """
        Gets the central pixel of a given list
        :return: central pixel
        """
        x_values = []
        y_values = []

        for p in pixels:
            x_values.append(p[0])
            y_values.append(p[1])

        x_center = round(statistics.median(x_values))
        y_center = round(statistics.median(y_values))

        return (x_center, y_center)

    def get_component_left(screen, pixels):
        """
        Gets the left pixel of a given list
        :return: left pixel
        """
        left = (screen.shape[1], 0)  # In this object element 0 means height, element 1 means width

        for p in pixels:
            if p[0] < left[0]:
                left = p

        return left

    def get_component_right(pixels):
        """
        Gets the right pixel of a given list
        :return: bottom pixel
        """
        right = (0, 0)

        for p in pixels:
            if p[0] > right[0]:
                right = p

        return right
