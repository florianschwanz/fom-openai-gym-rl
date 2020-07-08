import math

from argument_extractor import ArgumentExtractor
from environment_enum import Environment
from visual_analyzer import VisualAnalyzer
from visual_component import BreakoutBlockComponent
from visual_component import VisualComponent


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

    BLOCKS_X_MIN = 8
    BLOCKS_X_MAX = 150

    RED_BLOCKS_Y_MIN = 57
    RED_BLOCKS_Y_MAX = 62
    ORANGE_BLOCKS_Y_MIN = 63
    ORANGE_BLOCKS_Y_MAX = 68
    YELLOW_BLOCKS_Y_MIN = 69
    YELLOW_BLOCKS_Y_MAX = 74
    LIME_BLOCKS_Y_MIN = 75
    LIME_BLOCKS_Y_MAX = 80
    GREEN_BLOCKS_Y_MIN = 81
    GREEN_BLOCKS_Y_MAX = 86
    BLUE_BLOCKS_Y_MIN = 87
    BLUE_BLOCKS_Y_MAX = 92

    BALL_CENTER_Y_WHEN_PLAYED_BY_PLAYER_MIN = 180
    BALL_CENTER_Y_WHEN_PLAYED_BY_PLAYER_MAX = 184

    # Environments this reward shaper makes sense to use with
    ENVIRONMENTS = [
        Environment.BREAKOUT_V0,
        Environment.BREAKOUT_V4,
        Environment.BREAKOUT_DETERMINISTIC_V0,
        Environment.BREAKOUT_DETERMINISTIC_V4,
        Environment.BREAKOUT_NO_FRAMESKIP_V0,
        Environment.BREAKOUT_NO_FRAMESKIP_V4
    ]

    # Counters
    RED_BLOCKS_ON_SCREEN = 18
    ORANGE_BLOCKS_ON_SCREEN = 18
    YELLOW_BLOCKS_ON_SCREEN = 18
    LIME_BLOCKS_ON_SCREEN = 18
    GREEN_BLOCKS_ON_SCREEN = 18
    BLUE_BLOCKS_ON_SCREEN = 18

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

    def reset_reward_shaper(self):
        BreakoutRewardShaper.RED_BLOCKS_ON_SCREEN = 18
        BreakoutRewardShaper.ORANGE_BLOCKS_ON_SCREEN = 18
        BreakoutRewardShaper.YELLOW_BLOCKS_ON_SCREEN = 18
        BreakoutRewardShaper.LIME_BLOCKS_ON_SCREEN = 18
        BreakoutRewardShaper.GREEN_BLOCKS_ON_SCREEN = 18
        BreakoutRewardShaper.BLUE_BLOCKS_ON_SCREEN = 18

    def initialize_reward_shaper(func):
        def initialize_reward_shaper_and_call(self, *args, **kwargs):
            self.screen = ArgumentExtractor.extract_argument(kwargs, "screen", None)
            self.reward = ArgumentExtractor.extract_argument(kwargs, "reward", None)
            self.done = ArgumentExtractor.extract_argument(kwargs, "done", None)
            self.info = ArgumentExtractor.extract_argument(kwargs, "info", None)

            self.ball_pixels, \
            self.player_racket_pixels = BreakoutRewardShaper.extract_pixels(self, self.screen)

            self.pixels = VisualAnalyzer.extract_pixels(self.screen)
            self.colors = VisualAnalyzer.extract_colors(self.pixels)

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

    @check_environment
    @initialize_reward_shaper
    def reward_ball_hitting_upper_block(self, **kwargs):
        screen = ArgumentExtractor.extract_argument(kwargs, "screen", None)
        original_reward = ArgumentExtractor.extract_argument(kwargs, "reward", None)
        additional_reward = ArgumentExtractor.extract_argument(kwargs, "additional_reward", 0)

        # Pixel lists
        red_blocks_pixels = []
        orange_blocks_pixels = []
        yellow_blocks_pixels = []
        lime_blocks_pixels = []
        green_blocks_pixels = []
        blue_blocks_pixels = []

        if original_reward > 0:

            # Define relevant section of the screen
            section_x_min = self.WALL_LEFT_X_MAX + 1
            section_x_max = self.WALL_RIGHT_Y_MIN - 1
            section_y_min = self.RED_BLOCKS_Y_MIN
            section_y_max = self.BLUE_BLOCKS_Y_MAX

            # Define step size
            steps_x = 2
            steps_y = 6

            for x in range(section_x_min, section_x_max, steps_x):
                for y in range(section_y_min, section_y_max, steps_y):
                    coordinates = (x, y)
                    value = (screen[y][x][0], screen[y][x][1], screen[y][x][2])

                    if BreakoutRewardShaper.is_red_block_pixel(self, x, y, value):
                        red_blocks_pixels.append(coordinates)
                    elif BreakoutRewardShaper.is_orange_block_pixel(self, x, y, value):
                        orange_blocks_pixels.append(coordinates)
                    elif BreakoutRewardShaper.is_yellow_block_pixel(self, x, y, value):
                        yellow_blocks_pixels.append(coordinates)
                    elif BreakoutRewardShaper.is_lime_block_pixel(self, x, y, value):
                        lime_blocks_pixels.append(coordinates)
                    elif BreakoutRewardShaper.is_green_block_pixel(self, x, y, value):
                        green_blocks_pixels.append(coordinates)
                    elif BreakoutRewardShaper.is_blue_block_pixel(self, x, y, value):
                        blue_blocks_pixels.append(coordinates)

            red_blocks = BreakoutBlockComponent(red_blocks_pixels, self.screen)
            orange_blocks = BreakoutBlockComponent(orange_blocks_pixels, self.screen)
            yellow_blocks = BreakoutBlockComponent(yellow_blocks_pixels, self.screen)
            lime_blocks = BreakoutBlockComponent(lime_blocks_pixels, self.screen)
            green_blocks = BreakoutBlockComponent(green_blocks_pixels, self.screen)
            blue_blocks = BreakoutBlockComponent(blue_blocks_pixels, self.screen)

            if blue_blocks.num_blocks < BreakoutRewardShaper.BLUE_BLOCKS_ON_SCREEN:
                BreakoutRewardShaper.BLUE_BLOCKS_ON_SCREEN = blue_blocks.num_blocks
                return additional_reward * (1 / 6)
            elif green_blocks.num_blocks < BreakoutRewardShaper.GREEN_BLOCKS_ON_SCREEN:
                BreakoutRewardShaper.GREEN_BLOCKS_ON_SCREEN = green_blocks.num_blocks
                return additional_reward * (2 / 6)
            elif lime_blocks.num_blocks < BreakoutRewardShaper.LIME_BLOCKS_ON_SCREEN:
                BreakoutRewardShaper.LIME_BLOCKS_ON_SCREEN = lime_blocks.num_blocks
                return additional_reward * (3 / 6)
            elif yellow_blocks.num_blocks < BreakoutRewardShaper.YELLOW_BLOCKS_ON_SCREEN:
                BreakoutRewardShaper.YELLOW_BLOCKS_ON_SCREEN = yellow_blocks.num_blocks
                return additional_reward * (4 / 6)
            elif orange_blocks.num_blocks < BreakoutRewardShaper.ORANGE_BLOCKS_ON_SCREEN:
                BreakoutRewardShaper.ORANGE_BLOCKS_ON_SCREEN = orange_blocks.num_blocks
                return additional_reward * (5 / 6)
            elif red_blocks.num_blocks < BreakoutRewardShaper.RED_BLOCKS_ON_SCREEN:
                BreakoutRewardShaper.RED_BLOCKS_ON_SCREEN = red_blocks.num_blocks
                return additional_reward
            else:
                return 0
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

    def is_red_block_pixel(self, x, y, value):
        return value == self.RED \
               and self.BLOCKS_X_MIN <= x <= self.BLOCKS_X_MAX \
               and self.RED_BLOCKS_Y_MIN <= y <= self.RED_BLOCKS_Y_MAX

    def is_orange_block_pixel(self, x, y, value):
        return value == self.ORANGE \
               and self.BLOCKS_X_MIN <= x <= self.BLOCKS_X_MAX \
               and self.ORANGE_BLOCKS_Y_MIN <= y <= self.ORANGE_BLOCKS_Y_MAX

    def is_yellow_block_pixel(self, x, y, value):
        return value == self.YELLOW \
               and self.BLOCKS_X_MIN <= x <= self.BLOCKS_X_MAX \
               and self.YELLOW_BLOCKS_Y_MIN <= y <= self.YELLOW_BLOCKS_Y_MAX

    def is_lime_block_pixel(self, x, y, value):
        return value == self.LIME \
               and self.BLOCKS_X_MIN <= x <= self.BLOCKS_X_MAX \
               and self.LIME_BLOCKS_Y_MIN <= y <= self.LIME_BLOCKS_Y_MAX

    def is_green_block_pixel(self, x, y, value):
        return value == self.GREEN \
               and self.BLOCKS_X_MIN <= x <= self.BLOCKS_X_MAX \
               and self.GREEN_BLOCKS_Y_MIN <= y <= self.GREEN_BLOCKS_Y_MAX

    def is_blue_block_pixel(self, x, y, value):
        return value == self.BLUE \
               and self.BLOCKS_X_MIN <= x <= self.BLOCKS_X_MAX \
               and self.BLUE_BLOCKS_Y_MIN <= y <= self.BLUE_BLOCKS_Y_MAX
