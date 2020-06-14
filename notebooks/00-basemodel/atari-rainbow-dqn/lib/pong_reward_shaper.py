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
            self.center = PongRewardShaper.get_component_center(pixels)
            self.top = PongRewardShaper.get_component_top(screen, pixels)
            self.bottom = PongRewardShaper.get_component_bottom(pixels)
        else:
            self.visible = False
            self.center = ()
            self.top = ()
            self.bottom = ()


class PongRewardShaper():
    """
    Pixel-based reward shaper for Atari game Pong
    """

    # Colors contained in the game
    BLACK = (0, 0, 0)
    BROWN = (144, 72, 17)
    ORANGE = (213, 130, 74)
    GREEN = (92, 186, 92)
    LIGHTGREY = (236, 236, 236)

    # Important positions
    BALL_CENTER_X_WHEN_PLAYED_BY_PLAYER = 142
    BALL_CENTER_X_WHEN_PLAYED_BY_OPPONENT = 20

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
            environment_name = kwargs["environment_name"]

            if environment_name not in self.ENVIRONMENTS:
                raise Exception("Reward shaping method does match environment "
                                "(method:" + func.__name__ + ", environment:" + environment_name + ")")

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

            self.pixels = PongRewardShaper.extract_pixels(self.screen)
            # self.colors = PongRewardShaper.extract_colors(self.pixels)

            self.ball = Component(PongRewardShaper.get_ball_pixels(self.pixels), self.screen)
            self.player_racket = Component(PongRewardShaper.get_player_racket_pixels(self.pixels), self.screen)
            self.opponent_racket = Component(PongRewardShaper.get_opponent_racket_pixels(self.pixels), self.screen)

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
    def reward_player_racket_hits_ball(self, additional_reward=0.025):
        """
        Gives an additional reward if the player's racket hits the ball
        :return: shaped reward
        """

        if self.ball.visible and self.player_racket.visible \
                and self.ball.center[0] == self.BALL_CENTER_X_WHEN_PLAYED_BY_PLAYER \
                and self.player_racket.top[1] <= self.ball.center[1] <= self.player_racket.bottom[1]:
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
                and self.player_racket.top[1] <= self.ball.center[1] <= self.player_racket.bottom[1]:
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
            dist = abs(self.ball.center[1] - self.player_racket.center[1])
            additional_reward = round(((reward_max - reward_min) / (dist_min - dist_max) * dist + reward_max), 2)
            return additional_reward
        else:
            return 0

    @check_environment
    @initialize_reward_shaper
    def reward_player_racket_close_to_ball_quadractic(self, additional_reward=0.05):
        reward_max = math.sqrt(additional_reward)
        reward_min = 0

        dist_max = 160
        dist_min = 0

        if self.ball.visible and self.player_racket.visible:
            dist = abs(self.ball.center[1] - self.player_racket.center[1])
            additional_reward = round(((reward_max - reward_min) / (dist_min - dist_max) * dist + reward_max), 2)
            return math.pow(additional_reward, 2)
        else:
            return 0

    @check_environment
    @initialize_reward_shaper
    def reward_opponent_racket_hits_ball(self, additional_reward=-0.025):
        """
        Gives an additional reward if the oppponent's racket hits the ball
        :return: shaped reward
        """

        if self.ball.visible and self.opponent_racket.visible \
                and self.ball.center[0] == self.BALL_CENTER_X_WHEN_PLAYED_BY_OPPONENT \
                and self.opponent_racket.top[1] <= self.ball.center[1] <= self.opponent_racket.bottom[1]:
            return additional_reward
        else:
            return 0

    @check_environment
    @initialize_reward_shaper
    def reward_opponent_racket_covers_ball(self, additional_reward=-0.025):
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
    def reward_opponent_racket_close_to_ball_linear(self, additional_reward=-0.05):
        reward_max = additional_reward
        reward_min = 0

        dist_max = 160
        dist_min = 0

        if self.ball.visible and self.opponent_racket.visible:
            dist = abs(self.ball.center[1] - self.opponent_racket.center[1])
            additional_reward = round(((reward_max - reward_min) / (dist_min - dist_max) * dist + reward_max), 2)
            return additional_reward
        else:
            return 0

    @check_environment
    @initialize_reward_shaper
    def reward_opponent_racket_close_to_ball_quadractic(self, additional_reward=-0.05):
        reward_max = math.sqrt(additional_reward)
        reward_min = 0

        dist_max = 160
        dist_min = 0

        if self.ball.visible and self.opponent_racket.visible:
            dist = abs(self.ball.center[1] - self.opponent_racket.center[1])
            additional_reward = round(((reward_max - reward_min) / (dist_min - dist_max) * dist + reward_max), 2)
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

    def get_ball_pixels(pixels):
        """
        Gets all pixels that represent the ball by color
        :return: list of pixels representing the ball
        """

        BALL_COLOR = PongRewardShaper.LIGHTGREY

        ball_pixels = []

        for key, value in pixels.items():
            x = key[0]
            y = key[1]

            if value == BALL_COLOR \
                    and not (y >= 24 and y <= 33) \
                    and not (y >= 194 and y <= 209):
                ball_pixels.append(key)

        return ball_pixels

    def get_player_racket_pixels(pixels):
        """
        Gets all pixels that represent the player's racket by color
        :return: list of pixels representing the the player's racket
        """

        RACKET_COLOR = PongRewardShaper.GREEN

        racket_pixels = []

        for key, value in pixels.items():
            x = key[0]
            y = key[1]

            if value == RACKET_COLOR \
                    and (y >= 21):
                racket_pixels.append(key)

        return racket_pixels

    def get_opponent_racket_pixels(pixels):
        """
        Gets all pixels that represent the opponent's racket by color
        :return: list of pixels representing the the opponent's racket
        """

        RACKET_COLOR = PongRewardShaper.ORANGE

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

    def get_component_top(screen, pixels):
        """
        Gets the top pixel of a given list
        :return: top pixel
        """
        top = (0, screen.shape[0])  # In this object element 0 means height, element 1 means width

        for p in pixels:
            if p[1] < top[1]:
                top = p

        return top

    def get_component_bottom(pixels):
        """
        Gets the bottom pixel of a given list
        :return: bottom pixel
        """
        bottom = (0, 0)

        for p in pixels:
            if p[1] > bottom[1]:
                bottom = p

        return bottom
