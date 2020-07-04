import math

from lib.argument_extractor import ArgumentExtractor
from lib.environment_enum import Environment
from lib.visual_component import VisualComponent


class BreakoutRewardShaper():
    """
    Pixel-based reward shaper for Atari game Breakout
    """

    # Colors contained in the game
    BLACK = BACKGROUND_COLOR = (0, 0, 0)
    GREY = WALL_COLOR = (142, 142, 142)
    RED = BALL_COLOR = RACKET_COLOR = (200, 72, 72)
    ORANGE = (198, 108, 58)
    YELLOW = (180, 122, 48)
    LIME = (162, 162, 42)
    GREEN = (72, 160, 72)
    BLUE = (66, 72, 200)
    TEAL = (66, 158, 130)

    # Important positions
    LINE_RED_Y_MIN = 57
    LINE_RED_Y_MAX = 62
    PLAYER_RACKET_Y_MIN = 189
    PLAYER_RACKET_Y_MAX = 194
    WALL_LEFT_X_MAX = 7
    WALL_RIGHT_Y_MIN = 152
    WALL_TOP_Y_MAX = 31
    WALL_BOTTOM_Y_MAX = 195

    BALL_CENTER_Y_WHEN_PLAYED_BY_PLAYER_MIN = 180
    BALL_CENTER_Y_WHEN_PLAYED_BY_PLAYER_MAX = 184

    # Environments this reward shaper makes sense to use with
    ENVIRONMENTS = [
        Environment.BREAKOUT_V0,
        Environment.BREAKOUT_NO_FRAMESKIP_V0,
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

            self.ball_pixels, \
            self.player_racket_pixels = BreakoutRewardShaper.extract_pixels(self, self.screen)

            self.ball = VisualComponent(self.ball_pixels, self.screen)
            self.player_racket = VisualComponent(self.player_racket_pixels, self.screen)
            self.lives = self.info["ale.lives"]

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
    def reward_player_racket_hits_ball(self, **kwargs):
        """
        Gives an additional reward if the player's racket hits the ball
        :return: shaped reward
        """

        additional_reward = ArgumentExtractor.extract_argument(kwargs, "additional_reward", 0)

        if self.ball.visible and self.player_racket.visible \
                and self.BALL_CENTER_Y_WHEN_PLAYED_BY_PLAYER_MIN <= self.ball.center[1] <= \
                self.BALL_CENTER_Y_WHEN_PLAYED_BY_PLAYER_MAX \
                and self.player_racket.left[0] <= self.ball.center[0] <= self.player_racket.right[0]:
            return additional_reward
        else:
            return 0

    @check_environment
    @initialize_reward_shaper
    def reward_player_racket_covers_ball(self, **kwargs):
        """
        Gives an additional reward if the player's racket covers y-coordinate of the ball
        :return: shaped reward
        """

        additional_reward = ArgumentExtractor.extract_argument(kwargs, "additional_reward", 0)

        if self.ball.visible and self.player_racket.visible \
                and self.player_racket.left[0] <= self.ball.center[0] <= self.player_racket.right[0]:
            return additional_reward
        else:
            return 0

    @check_environment
    @initialize_reward_shaper
    def reward_player_racket_close_to_ball_linear(self, **kwargs):

        additional_reward = ArgumentExtractor.extract_argument(kwargs, "additional_reward", 0)

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
    def reward_player_racket_close_to_ball_quadratic(self, **kwargs):
        additional_reward = ArgumentExtractor.extract_argument(kwargs, "additional_reward", 0)

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

    def extract_pixels(self, screen):
        """
        Extracts pixels from a screen
        :return: a dictionary having coordinates as key, and rgb values as value
        """

        ball_pixels = []
        player_racket_pixels = []

        screen_height = screen.shape[0]  # y-axis starting from top-left corner
        screen_width = screen.shape[1]  # x-axis starting from top-left corner

        # Define relevant section of the screen
        section_x_min = self.WALL_LEFT_X_MAX + 1
        section_x_max = self.WALL_RIGHT_Y_MIN - 1
        section_y_min = self.WALL_TOP_Y_MAX + 1
        section_y_max = self.WALL_BOTTOM_Y_MAX - 1

        # Define step size
        steps_x = 2
        steps_y = 4

        for x in range(section_x_min, section_x_max, steps_x):
            for y in range(section_y_min, section_y_max, steps_y):
                coordinates = (x, y)
                value = (screen[y][x][0], screen[y][x][1], screen[y][x][2])

                if BreakoutRewardShaper.is_background_pixel(self, x, y, value):
                    pass
                elif BreakoutRewardShaper.is_ball_pixel(self, x, y, value):
                    ball_pixels.append(coordinates)
                elif BreakoutRewardShaper.is_player_racket_pixel(self, x, y, value):
                    player_racket_pixels.append(coordinates)

        return ball_pixels, player_racket_pixels

    def is_background_pixel(self, x, y, value):
        return value == self.BACKGROUND_COLOR

    def is_wall_pixel(self, x, y, value):
        return value == self.WALL_COLOR

    def is_ball_pixel(self, x, y, value):
        return value == self.BALL_COLOR \
               and not (y >= self.LINE_RED_Y_MIN and y <= self.LINE_RED_Y_MAX) \
               and not (y >= self.PLAYER_RACKET_Y_MIN and y <= self.PLAYER_RACKET_Y_MAX)

    def is_player_racket_pixel(self, x, y, value):
        return value == self.RACKET_COLOR \
               and not (y >= self.LINE_RED_Y_MIN and y <= self.LINE_RED_Y_MAX) \
               and (y >= self.PLAYER_RACKET_Y_MIN and y <= self.PLAYER_RACKET_Y_MAX)
