from enum import Enum


class RewardShape(Enum):
    PONG_PLAYER_RACKET_HITS_BALL = "pong-player-racket-hits-ball",
    PONG_PLAYER_RACKET_COVERS_BALL = "pong-player-racket-covers-ball",
    PONG_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR = "pong-player-racket-close-to-ball-linear",
    PONG_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC = "pong-player-racket-close-to-ball-quadratic",

    PONG_OPPONENT_RACKET_HITS_BALL = "pong-opponent-racket-hits-ball",
    PONG_OPPONENT_RACKET_COVERS_BALL = "pong-opponent-racket-hits-ball",
    PONG_OPPONENT_RACKET_CLOSE_TO_BALL_LINEAR = "pong-opponent-racket-close-to-ball-linear",
    PONG_OPPONENT_RACKET_CLOSE_TO_BALL_QUADRATIC = "pong-opponent-racket-close-to-ball-quadratic",
