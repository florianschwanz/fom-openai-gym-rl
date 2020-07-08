import statistics

class BreakoutBlockComponent:
    """
    Represents breakout blocks of a certain color
    """

    def __init__(self, pixels, screen):
        if len(pixels) == 0:
            self.num_blocks = 0
        else:
            self.num_blocks = int(len(pixels) / 4)

class VisualComponent:
    """
    Represents a distinct visual entity on the screen
    """

    def __init__(self, pixels, screen):
        if len(pixels) > 0:
            self.visible = True
            self.center = VisualComponent.get_component_center(pixels)
            self.top = VisualComponent.get_component_top(screen, pixels)
            self.bottom = VisualComponent.get_component_bottom(pixels)
            self.left = VisualComponent.get_component_left(screen, pixels)
            self.right = VisualComponent.get_component_right(pixels)
        else:
            self.visible = False
            self.center = ()
            self.top = ()
            self.bottom = ()
            self.left = ()
            self.right = ()

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
