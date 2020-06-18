from enum import Enum


class Environment(Enum):
    PONG_v0 = 'Pong-v0'
    PONG_v4 = 'Pong-v4'
    PONG_DETERMINISTIC_v0 = 'PongDeterministic-v0'
    PONG_DETERMINISTIC_v4 = 'PongDeterministic-v4'
    PONG_NO_FRAMESKIP_v0 = 'PongNoFrameskip-v0'
    PONG_NO_FRAMESKIP_v4 = 'PongNoFrameskip-v4'

    BREAKOUT_V0 = 'Breakout-v0'

    SPACE_INVADERS_V0 = 'SpaceInvaders-v0'

    CART_POLE_v0 = 'CartPole-v0'

    FREEWAY_V0 = 'Freeway-v0'
    GRAVITAR_V0 = 'Gravitar-v0'
    MONTEZUMAS_REVENGE_V0 = 'MontezumaRevenge-v0'
    PITFALL_V0 = 'Pitfall-v0'
    PRIVATE_EYE_V0 = 'PrivateEye-v0'
    SOLARIS_V0 = 'Solaris-v0'
    VENTURE_V0 = 'Venture-v0'
