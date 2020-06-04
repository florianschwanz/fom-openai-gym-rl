from enum import Enum


class RewardShape(Enum):
    PONG_PLAYER_RACKET_CENTER_ON_BALL = "pong-player-racket-center-on-ball",
    PONG_PLAYER_RACKET_CLOSE_TO_BALL = "pong-player-racket-close-to-ball",
    PONG_PLAYER_RACKET_PROXIMITY_TO_BALL_LINEAR = "pong-player-racket-proximity-to-ball-linear",
    PONG_PLAYER_RACKET_PROXIMITY_TO_BALL_QUADRATIC = "pong-player-racket-proximity-to-ball-quadratic"
    PONG_OPPONENT_RACKET_CLOSE_TO_BALL_QUADRATIC = "pong-opponent-racket-close-to-ball-quadratic"
