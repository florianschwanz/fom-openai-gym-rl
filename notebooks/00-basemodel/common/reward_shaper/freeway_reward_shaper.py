from environment_enum import Environment
from visual_component import VisualComponent


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

    # Environments this reward shaper makes sense to use with
    ENVIRONMENTS = [
        Environment.FREEWAY_V0,
        Environment.FREEWAY_NO_FRAMESKIP_V0
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

            self.player_chicken_pixels = self.extract_pixels(self.screen)

            self.player_chicken = VisualComponent(self.player_chicken_pixels, self.screen)
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
            if (self.chicken.visible):
                print("DEBUG"
                      + " chicken " + str(self.chicken.center))
            else:
                print("DEBUG "
                      + " no chicken")
            return func(self, *args, **kwargs)

        return debug_positions_and_call

    @check_environment
    @initialize_reward_shaper
    def reward_chicken_vertical_position(self, additional_reward=0.5):
        """
        Gives an additional reward if the player's racket hits the ball
        :return: shaped reward
        """

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
        Extracts pixels from an screen
        :return: a dictionary having coordinates as key, and rgb values as value
        """

        player_chicken_pixels = []

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

        return player_chicken_pixels

    def is_background_pixel(self, x, y, value):
        return value == self.BACKGROUND_COLOR

    def is_player_chicken_pixel(self, x, y, value):
        return value == self.CHICKEN_COLOR \
               and x >= self.PLAYER_1_CHICKEN_X_MIN and x <= self.PLAYER_1_CHICKEN_X_MAX \
               and y != self.MEDIAN_STRIP_TOP_Y \
               and y != self.MEDIAN_STRIP_BOTTOM_Y
