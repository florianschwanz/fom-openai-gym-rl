import time

from argument_extractor import ArgumentExtractor
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

    CAR_CENTER_Y = [
        CAR_1_CENTER_Y,
        CAR_2_CENTER_Y,
        CAR_3_CENTER_Y,
        CAR_4_CENTER_Y,
        CAR_5_CENTER_Y,
        CAR_6_CENTER_Y,
        CAR_7_CENTER_Y,
        CAR_8_CENTER_Y,
        CAR_9_CENTER_Y,
        CAR_10_CENTER_Y,
    ]

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

    def init(self, *args, **kwargs):
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
        self.cars = [
            VisualComponent(self.car_1_pixels, self.screen),
            VisualComponent(self.car_2_pixels, self.screen),
            VisualComponent(self.car_3_pixels, self.screen),
            VisualComponent(self.car_4_pixels, self.screen),
            VisualComponent(self.car_5_pixels, self.screen),
            VisualComponent(self.car_6_pixels, self.screen),
            VisualComponent(self.car_7_pixels, self.screen),
            VisualComponent(self.car_8_pixels, self.screen),
            VisualComponent(self.car_9_pixels, self.screen),
            VisualComponent(self.car_10_pixels, self.screen)
        ]

        self.lives = self.info["ale.lives"]

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
            self.cars = [
                VisualComponent(self.car_1_pixels, self.screen),
                VisualComponent(self.car_2_pixels, self.screen),
                VisualComponent(self.car_3_pixels, self.screen),
                VisualComponent(self.car_4_pixels, self.screen),
                VisualComponent(self.car_5_pixels, self.screen),
                VisualComponent(self.car_6_pixels, self.screen),
                VisualComponent(self.car_7_pixels, self.screen),
                VisualComponent(self.car_8_pixels, self.screen),
                VisualComponent(self.car_9_pixels, self.screen),
                VisualComponent(self.car_10_pixels, self.screen)
            ]

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

            if (self.player_chicken.visible):
                print("DEBUG chicken " + str(self.player_chicken.center))

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
    def reward_distance_walked(self, **kwargs):
        """
        Gives an additional reward if the chicken has a huge distance to a car that can hit on the lane it stands on
        :return: shaped reward
        """

        additional_reward = ArgumentExtractor.extract_argument(kwargs, "additional_reward", 0)

        reward_max = additional_reward
        reward_min = - additional_reward

        dist_max = 0
        dist_min = self.screen.shape[0] # Screen height

        distance_walked = FreewayRewardShaper.get_distance_walked()

        if self.player_chicken.visible:
            m = ((reward_max - reward_min) / (dist_max - dist_min))
            n = reward_min - (m * dist_min)
            additional_reward = m * distance_walked + n
            return additional_reward
        else:
            return 0

    @check_environment
    @initialize_reward_shaper
    def reward_distance_to_car(self, **kwargs):
        """
        Gives an additional reward if the chicken has a huge distance to a car that can hit on the lane it stands on
        :return: shaped reward
        """

        additional_reward = ArgumentExtractor.extract_argument(kwargs, "additional_reward", 0)

        reward_max = additional_reward
        reward_min = - additional_reward

        dist_max = self.screen.shape[1]  # Screen width
        dist_min = 0

        distance_to_car = FreewayRewardShaper.get_distance_to_car(self)

        if self.player_chicken.visible:
            m = ((reward_max - reward_min) / (dist_max - dist_min))
            n = reward_min - (m * dist_min)
            additional_reward = m * distance_to_car + n
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

    def get_chicken_lane(self):
        """
        Determines which lane the chicken is on starting with the bottom lane being 1
        :return: lane the chicken is on
        """
        if self.player_chicken.visible:
            if (169 <= self.player_chicken.center[1] <= 184):
                return 1
            elif (153 <= self.player_chicken.center[1] <= 168):
                return 2
            elif (137 <= self.player_chicken.center[1] <= 152):
                return 3
            elif (119 <= self.player_chicken.center[1] <= 136):
                return 4
            elif (103 <= self.player_chicken.center[1] <= 120):
                return 5

            elif (85 <= self.player_chicken.center[1] <= 100):
                return 6
            elif (69 <= self.player_chicken.center[1] <= 84):
                return 7
            elif (53 <= self.player_chicken.center[1] <= 68):
                return 8
            elif (37 <= self.player_chicken.center[1] <= 52):
                return 9
            elif (21 <= self.player_chicken.center[1] <= 36):
                return 10
            else:
                return None
        else:
            return None

    def get_distance_walked(self):
        return self.screen.shape[0] - self.player_chicken.center[1]

    def get_distance_to_car(self):
        """
        Calculates the distance the next car has to drive in order to hit the chicken
        :return:
        """
        chicken_lane = FreewayRewardShaper.get_chicken_lane(self)

        # Check if chicken is on a lane
        if chicken_lane != None:
            car_in_lane = self.cars[chicken_lane - 1]
            screen_width = self.screen.shape[1]

            # Check if car is current lane is visible
            if car_in_lane.visible:
                chicken_center_x = self.player_chicken.center[0]
                car_in_lane_center_x = car_in_lane.center[0]

                # Check if chicken are collide
                if abs(chicken_center_x - car_in_lane_center_x) < 6:
                    return 0

                # Check if chicken is in first five lanes
                if chicken_lane <= 5:
                    # Check if car is left of chicken
                    if car_in_lane_center_x < chicken_center_x:
                        return chicken_center_x - car_in_lane_center_x
                    # Check if car is right of chicken
                    else:
                        return screen_width + chicken_center_x - car_in_lane_center_x

                # Check if chicken is in second five lanes
                elif chicken_lane >= 6:
                    # Check if car is left of chicken
                    if car_in_lane_center_x < chicken_center_x:
                        return screen_width + car_in_lane_center_x - chicken_center_x
                    # Check if car is right of chicken
                    else:
                        return car_in_lane_center_x - chicken_center_x
            else:
                if chicken_lane <= 5:
                    return self.player_chicken.center[0]
                elif chicken_lane >= 6:
                    return screen_width - self.player_chicken.center[0]
        else:
            return 0
