import math

from argument_extractor import ArgumentExtractor
from environment_enum import Environment
from visual_component import VisualComponent


class PongRewardShaper():
    """
    Pixel-based reward shaper for Atari game Pong
    """

    # Colors contained in the game
    BLACK = (0, 0, 0)
    BROWN = BACKGROUND_COLOR = (144, 72, 17)
    ORANGE = OPPONENT_RACKET_COLOR = (213, 130, 74)
    GREEN = PLAYER_RACKET_COLOR = (92, 186, 92)
    LIGHTGREY = BALL_COLOR = (236, 236, 236)

    # Important positions
    BALL_CENTER_X_WHEN_PLAYED_BY_PLAYER = 142
    BALL_CENTER_X_WHEN_PLAYED_BY_OPPONENT = 20
    GREY_BAR_TOP_Y_MIN = 24
    GREY_BAR_TOP_Y_MAX = 33
    GREY_BAR_BOTTOM_Y_MIN = 194
    GREY_BAR_BOTTOM_Y_MAX = 209
    SCORE_Y_MAX = 21

    # Environments this reward shaper makes sense to use with
    ENVIRONMENTS = [
        Environment.PONG_v0,
        Environment.PONG_v4,
        Environment.PONG_DETERMINISTIC_v0,
        Environment.PONG_DETERMINISTIC_v4,
        Environment.PONG_NO_FRAMESKIP_v0,
        Environment.PONG_NO_FRAMESKIP_v4
    ]

    def check_environment(func):
        def check_environment_and_call(self, *args, **kwargs):
            """Checks if reward shaping is done on a matching environment"""
            environment = ArgumentExtractor.extract_argument(kwargs, "environment", None)

            if environment not in self.ENVIRONMENTS:
                raise Exception("Reward shaping method does match environment "
                                "(method:" + func.__name__ + ", environment:" + environment.value + ")")

            return func(self, *args, **kwargs)

        return check_environment_and_call

    def initialize_reward_shaper(func):
        def initialize_reward_shaper_and_call(self, *args, **kwargs):
            self.screen = ArgumentExtractor.extract_argument(kwargs, "screen", None)
            self.reward = ArgumentExtractor.extract_argument(kwargs, "reward", None)
            self.done = ArgumentExtractor.extract_argument(kwargs, "done", None)
            self.info = ArgumentExtractor.extract_argument(kwargs, "info", None)

            self.ball_pixels, \
            self.player_racket_pixels, \
            self.opponent_racket_pixels = self.extract_pixels(self.screen)

            self.ball = VisualComponent(self.ball_pixels, self.screen)
            self.player_racket = VisualComponent(self.player_racket_pixels, self.screen)
            self.opponent_racket = VisualComponent(self.opponent_racket_pixels, self.screen)
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
                      + " opponent " + str(self.opponent_racket.center)
                      + " ball " + str(self.ball.center))
            else:
                print("DEBUG "
                      + " player " + str(self.player_racket.center)
                      + " opponent " + str(self.opponent_racket.center)
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
                and self.ball.center[0] == self.BALL_CENTER_X_WHEN_PLAYED_BY_PLAYER \
                and self.player_racket.top[1] <= self.ball.center[1] <= self.player_racket.bottom[1]:
            return additional_reward
        else:
            return 0

    @check_environment
    @initialize_reward_shaper
    def reward_player_racket_covers_ball(self, **kwargs):
        additional_reward = ArgumentExtractor.extract_argument(kwargs, "additional_reward", 0)

        """
        Gives an additional reward if the player's racket covers y-coordinate of the ball
        :return: shaped reward
        """

        if self.ball.visible and self.player_racket.visible \
                and self.player_racket.top[1] <= self.ball.center[1] <= self.player_racket.bottom[1]:
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
            dist = abs(self.ball.center[1] - self.player_racket.center[1])
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
            dist = abs(self.ball.center[1] - self.player_racket.center[1])
            additional_reward = round(((reward_max - reward_min) / (dist_min - dist_max) * dist + reward_min), 4)
            return math.pow(additional_reward, 2)
        else:
            return 0

    @check_environment
    @initialize_reward_shaper
    def reward_opponent_racket_hits_ball(self, **kwargs):
        """
        Gives an additional reward if the oppponent's racket hits the ball
        :return: shaped reward
        """

        additional_reward = ArgumentExtractor.extract_argument(kwargs, "additional_reward", 0)

        if self.ball.visible and self.opponent_racket.visible \
                and self.ball.center[0] == self.BALL_CENTER_X_WHEN_PLAYED_BY_OPPONENT \
                and self.opponent_racket.top[1] <= self.ball.center[1] <= self.opponent_racket.bottom[1]:
            return additional_reward
        else:
            return 0

    @check_environment
    @initialize_reward_shaper
    def reward_opponent_racket_covers_ball(self, **kwargs):
        additional_reward = ArgumentExtractor.extract_argument(kwargs, "additional_reward", 0)

        """
        Gives an additional reward if the opponent's racket covers y-coordinate of the ball
        :return: shaped reward
        """

        if self.ball.visible and self.opponent_racket.visible \
                and self.opponent_racket.top[1] <= self.ball.center[1] <= self.opponent_racket.bottom[1]:
            return additional_reward
        else:
            return 0

    @check_environment
    @initialize_reward_shaper
    def reward_opponent_racket_close_to_ball_linear(self, **kwargs):
        additional_reward = ArgumentExtractor.extract_argument(kwargs, "additional_reward", 0)

        reward_max = additional_reward
        reward_min = 0

        dist_max = 160
        dist_min = 0

        if self.ball.visible and self.opponent_racket.visible:
            dist = abs(self.ball.center[1] - self.opponent_racket.center[1])
            additional_reward = round(((reward_max - reward_min) / (dist_min - dist_max) * dist + reward_max), 4)
            return additional_reward
        else:
            return 0

    @check_environment
    @initialize_reward_shaper
    def reward_opponent_racket_close_to_ball_quadratic(self, **kwargs):
        additional_reward = ArgumentExtractor.extract_argument(kwargs, "additional_reward", 0)

        reward_max = math.sqrt(additional_reward)
        reward_min = 0

        dist_max = 160
        dist_min = 0

        if self.ball.visible and self.opponent_racket.visible:
            dist = abs(self.ball.center[1] - self.opponent_racket.center[1])
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
        opponent_racket_pixels = []

        screen_height = screen.shape[0]  # y-axis starting from top-left corner
        screen_width = screen.shape[1]  # x-axis starting from top-left corner

        # Define relevant section of the screen
        section_x_min = 0
        section_x_max = screen_width
        section_y_min = self.GREY_BAR_TOP_Y_MAX + 1
        section_y_max = self.GREY_BAR_BOTTOM_Y_MIN - 1

        # Define step size
        steps_x = 1
        steps_y = 2

        for x in range(section_x_min, section_x_max, steps_x):
            for y in range(section_y_min, section_y_max, steps_y):
                coordinates = (x, y)
                value = (screen[y][x][0], screen[y][x][1], screen[y][x][2])

                if PongRewardShaper.is_background_pixel(self, x, y, value):
                    pass
                elif PongRewardShaper.is_ball_pixel(self, x, y, value):
                    ball_pixels.append(coordinates)
                elif PongRewardShaper.is_player_racket_pixel(self, x, y, value):
                    player_racket_pixels.append(coordinates)
                elif PongRewardShaper.is_opponent_racket_pixel(self, x, y, value):
                    opponent_racket_pixels.append(coordinates)

        return ball_pixels, player_racket_pixels

    def is_background_pixel(self, x, y, value):
        return value == self.BACKGROUND_COLOR

    def is_ball_pixel(self, x, y, value):
        return value == self.BALL_COLOR \
               and not (y >= self.GREY_BAR_TOP_Y_MIN and y <= self.GREY_BAR_TOP_Y_MAX) \
               and not (y >= self.GREY_BAR_BOTTOM_Y_MIN and y <= self.GREY_BAR_BOTTOM_Y_MAX)

    def is_player_racket_pixel(self, x, y, value):
        return value == self.PLAYER_RACKET_COLOR \
               and (y >= self.SCORE_Y_MAX)

    def is_opponent_racket_pixel(self, x, y, value):
        return value == self.OPPONENT_RACKET_COLOR \
               and (y >= self.SCORE_Y_MAX)
