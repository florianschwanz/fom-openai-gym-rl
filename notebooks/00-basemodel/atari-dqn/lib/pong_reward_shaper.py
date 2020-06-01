import statistics


class PongRewardShaper:
    BLACK = (0, 0, 0)
    BROWN = (144, 72, 17)
    ORANGE = (213, 130, 74)
    GREEN = (92, 186, 92)
    LIGHTGREY = (236, 236, 236)

    def __init__(self, observation, reward, done, info):
        """
        Constructor
        :param observation: observation of the game
        :param reward: original reward
        :param done: information if game round is finished
        :param info: additional information
        """

        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info

        self.pixels = PongRewardShaper.extract_pixels(observation)
        self.colors = PongRewardShaper.extract_colors(self.pixels)
        self.ball_pixels = PongRewardShaper.get_ball_pixels(self.pixels)
        self.racket_pixels = PongRewardShaper.get_racket_pixels(self.pixels)

        self.ball_center = None
        self.racket_top = None
        self.racket_bottom = None
        self.racket_center = None

        if len(self.ball_pixels) > 0:
            self.ball_center = PongRewardShaper.get_component_center(self.ball_pixels)

        if len(self.racket_pixels) > 0:
            self.racket_top = PongRewardShaper.get_component_top(self, self.racket_pixels)
            self.racket_bottom = PongRewardShaper.get_component_bottom(self.racket_pixels)
            self.racket_center = PongRewardShaper.get_component_center(self.racket_pixels)

    def reward_center_ball(self, additional_reward=0.5):
        """
        Gives an additional reward if the player's racket is placed on the same y-coordinate as the ball
        :return: shaped reward
        """

        if self.ball_center is not None and self.racket_center is not None \
                and self.ball_center[1] == self.racket_center[1]:
            return self.reward + additional_reward
        else:
            return self.reward

    def reward_close_to_ball(self, additional_reward=0.25):
        """
        Gives an additional reward if the player's racket covers y-coordinate of the ball
        :return: shaped reward
        """

        if self.ball_center is not None and self.racket_top is not None and self.racket_bottom is not None \
                and self.racket_top[1] <= self.ball_center[1] <= self.racket_bottom[1]:
            return self.reward + additional_reward
        else:
            return self.reward

    def reward_vertical_proximity_to_ball(self, max_additional_reward=0.5):
        reward_max = max_additional_reward
        reward_min = 0

        dist_max = 160
        dist_min = 0

        if self.ball_center is not None and self.racket_center is not None:
            dist = abs(self.ball_center[1] - self.racket_center[1])
            additional_reward = round(((reward_max - reward_min) / (dist_min - dist_max) * dist + reward_max), 2)
            return self.reward + additional_reward
        else:
            return self.reward

    def extract_pixels(observation):
        """
        Extracts pixels from an observation
        :return: a dictionary having coordinates as key, and rgb values as value
        """
        pixels = {}

        observation_height = observation.shape[0]  # y-axis starting from top-left corner
        observation_width = observation.shape[1]  # x-axis starting from top-left corner
        # observation_dim = observation.shape[2]

        for h in range(observation_height):
            for w in range(observation_width):
                coordinates = (w, h)  # Flip with and height here to match regular x-y syntax
                value = (observation[h][w][0], observation[h][w][1], observation[h][w][2])

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

    def get_racket_pixels(pixels):
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

    def get_component_top(self, pixels):
        """
        Gets the top pixel of a given list
        :return: top pixel
        """
        top = ()
        min_y = self.observation.shape[0]

        for p in pixels:
            if p[1] < min_y:
                top = p

        return top

    def get_component_bottom(pixels):
        """
        Gets the bottom pixel of a given list
        :return: bottom pixel
        """
        bottom = ()
        max_y = 0

        for p in pixels:
            if p[1] > max_y:
                bottom = p

        return bottom
