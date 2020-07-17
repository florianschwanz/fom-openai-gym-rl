import math

from argument_extractor import ArgumentExtractor
from environment_enum import Environment
from visual_analyzer import VisualAnalyzer
from visual_component import VisualComponent


class MsPacmanRewardShaper():
    """
    Pixel-based reward shaper for Atari game Pong
    """

    # Colors contained in the game
    BLACK = HUD_COLOR = (0, 0, 0)
    DARK_BLUE = BACKGROUND_COLOR = (200, 72, 72)
    ROSE = FOOD_COLOR = WALL_COLOR = (228, 111, 111)
    YELLOW = PACMAN_COLOR = (210, 164, 74)
    LIME = (187, 187, 53)

    RED = SHADOW_COLOR = BLINKY_COLOR = (184, 50, 50)  # Follows Pac Man
    PINK = SPEEDY_COLOR = PINKY_COLOR = (198, 89, 179)  # Surrounds Pac Man
    LIGHT_BLUE = BASHFUL_COLOR = INKY_COLOR = (0, 28, 136)  # Randomly mimics one of the other three
    OCHER = POKEY_COLOR = CLYDE_COLOR = (195, 144, 61)  # Acts stupid

    # Important positions
    HUD_TOP_Y_MAX = 1
    HUD_BOTTOM_Y_MIN = 172

    # Environments this reward shaper makes sense to use with
    ENVIRONMENTS = [
        Environment.MS_PACMAN_V0,
        Environment.MS_PACMAN_V4,
        Environment.MS_PACMAN_DETERMINISTIC_V0,
        Environment.MS_PACMAN_DETERMINISTIC_V4,
        Environment.MS_PACMAN_NO_FRAMESKIP_V0,
        Environment.MS_PACMAN_NO_FRAMESKIP_V4
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

            pixels = VisualAnalyzer.extract_pixels(self.screen)
            colors = VisualAnalyzer.extract_colors(pixels)

            self.ms_pacman_pixels, \
            self.food_pixels, \
            self.blinky_pixels, \
            self.pinky_pixels, \
            self.inky_pixels, \
            self.clyde_pixels = self.extract_pixels(self.screen)

            self.ms_pacman = None
            self.food = None
            self.blinky = None
            self.pinky = None
            self.inky = None
            self.clyde = None

            if len(self.ms_pacman_pixels) > 0:
                self.ms_pacman = VisualComponent(self.ms_pacman_pixels, self.screen)
            self.food = VisualComponent(self.ms_pacman_pixels, self.screen)
            if len(self.blinky_pixels) > 0:
                self.blinky = VisualComponent(self.blinky_pixels, self.screen)
            if len(self.pinky_pixels) > 0:
              self.pinky = VisualComponent(self.pinky_pixels, self.screen)
            if len(self.inky_pixels) > 0:
             self.inky = VisualComponent(self.inky_pixels, self.screen)
            if len(self.clyde_pixels) > 0:
                self.clyde = VisualComponent(self.clyde_pixels, self.screen)
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
            print("DEBUG"
                  + " ms pacman " + str(self.ms_pacman.center))

            return func(self, *args, **kwargs)

        return debug_positions_and_call

    @check_environment
    @initialize_reward_shaper
    def reward_ms_pacman_far_from_enemy(self, **kwargs):
        """
        Gives an additional reward if Ms Pacman is far from next enemy
        :return: shaped reward
        """

        additional_reward = ArgumentExtractor.extract_argument(kwargs, "additional_reward", 0)

        distances = []

        distance_to_blinky = MsPacmanRewardShaper.distance(self.ms_pacman, self.blinky)
        distance_to_pinky = MsPacmanRewardShaper.distance(self.ms_pacman, self.pinky)
        distance_to_inky = MsPacmanRewardShaper.distance(self.ms_pacman, self.inky)
        distance_to_clyde = MsPacmanRewardShaper.distance(self.ms_pacman, self.clyde)

        if distance_to_blinky != None:
            distances.append(distance_to_blinky)
        if distance_to_pinky != None:
            distances.append(distance_to_pinky)
        if distance_to_inky != None:
            distances.append(distance_to_inky)
        if distance_to_clyde != None:
            distances.append(distance_to_clyde)

        if len(distances) > 0:
            distance_min = min(distances)
            distance_max = min(distances)

            reward_max = additional_reward
            reward_min = 0

            dist_max = math.sqrt(2 * math.pow(150, 2))
            dist_min = 0

            additional_reward = round(((reward_max - reward_min) / (dist_min - dist_max) * distance_min + reward_max), 4)
            return additional_reward
        else:
            return 0

    def extract_pixels(self, screen):
        """
        Extracts pixels from a screen
        :return: a dictionary having coordinates as key, and rgb values as value
        """

        ms_pacman_pixels = []
        food_pixels = []
        blinky_pixels = []
        pinky_pixels = []
        inky_pixels = []
        clyde_pixels = []

        screen_height = screen.shape[0]  # y-axis starting from top-left corner
        screen_width = screen.shape[1]  # x-axis starting from top-left corner

        # Define relevant section of the screen
        section_x_min = 0
        section_x_max = screen_width
        section_y_min = self.HUD_TOP_Y_MAX + 1
        section_y_max = self.HUD_BOTTOM_Y_MIN - 1

        # Define step size
        steps_x = 1
        steps_y = 1

        for x in range(section_x_min, section_x_max, steps_x):
            for y in range(section_y_min, section_y_max, steps_y):
                coordinates = (x, y)
                value = (screen[y][x][0], screen[y][x][1], screen[y][x][2])

                if MsPacmanRewardShaper.is_background_pixel(self, x, y, value):
                    pass
                elif MsPacmanRewardShaper.is_ms_pacman_pixel(self, x, y, value):
                    ms_pacman_pixels.append(coordinates)
                elif MsPacmanRewardShaper.is_food_pixel(self, x, y, value):
                    food_pixels.append(coordinates)
                elif MsPacmanRewardShaper.is_blinky_pixel(self, x, y, value):
                    blinky_pixels.append(coordinates)
                elif MsPacmanRewardShaper.is_pinky_pixel(self, x, y, value):
                    pinky_pixels.append(coordinates)
                elif MsPacmanRewardShaper.is_inky_pixel(self, x, y, value):
                    inky_pixels.append(coordinates)
                elif MsPacmanRewardShaper.is_clyde_pixel(self, x, y, value):
                    clyde_pixels.append(coordinates)

        return ms_pacman_pixels, food_pixels, blinky_pixels, pinky_pixels, inky_pixels, clyde_pixels

    def distance(one, two):
        if one != None and two != None:
            delta_x = abs(one.center[0] - two.center[0])
            delta_y = abs(one.center[1] - two.center[1])
            return math.sqrt(math.pow(delta_x, 2) + math.pow(delta_y, 2))
        else:
            return None

    def is_background_pixel(self, x, y, value):
        return value == self.BACKGROUND_COLOR

    def is_ms_pacman_pixel(self, x, y, value):
        return value == self.PACMAN_COLOR

    def is_food_pixel(self, x, y, value):
        return value == self.FOOD_COLOR

    def is_blinky_pixel(self, x, y, value):
        return value == self.BLINKY_COLOR

    def is_pinky_pixel(self, x, y, value):
        return value == self.PINKY_COLOR

    def is_inky_pixel(self, x, y, value):
        return value == self.INKY_COLOR

    def is_clyde_pixel(self, x, y, value):
        return value == self.CLYDE_COLOR

    def is_hud_pixel(self, x, y, value):
        return value == self.HUD_COLOR
