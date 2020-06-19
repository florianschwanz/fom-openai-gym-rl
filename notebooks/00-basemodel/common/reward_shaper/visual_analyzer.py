
class VisualAnalyzer:
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