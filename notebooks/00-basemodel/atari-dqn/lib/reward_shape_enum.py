from enum import Enum


class RewardShape(Enum):
    PONG_CENTER_RACKET_ON_BALL = "pong-center-racket-on-ball",
    PONG_RACKET_CLOSE_TO_BALL = "pong-racket-close-to-ball",
    PONG_PROXIMITY_TO_BALL_LINEAR = "pong-proximity-to-ball-linear",
    PONG_PROXIMITY_TO_BALL_QUADRATIC = "pong-proximity-to-ball-quadratic"
