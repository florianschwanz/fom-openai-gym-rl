from enum import Enum


class Environment(Enum):
    PONG_v0 = 'Pong-v0'
    PONG_v4 = 'Pong-v4'
    PONG_DETERMINISTIC_v0 = 'PongDeterministic-v0'
    PONG_DETERMINISTIC_v4 = 'PongDeterministic-v4'
    PONG_NO_FRAMESKIP_v0 = 'PongNoFrameskip-v0'
    PONG_NO_FRAMESKIP_v4 = 'PongNoFrameskip-v4'

    BREAKOUT_V0 = 'Breakout-v0'
    BREAKOUT_NO_FRAMESKIP_V0 = 'BreakoutNoFrameskip-v0'

    SPACE_INVADERS_V0 = 'SpaceInvaders-v0'
    SPACE_INVADERS_NO_FRAMESKIP_V0 = 'SpaceInvadersNoFrameskip-v0'

    CART_POLE_v0 = 'CartPole-v0'
    CART_POLE_NO_FRAMESKIP_v0 = 'CartPole-v0'

    FREEWAY_V0 = 'Freeway-v0'
    FREEWAY_NO_FRAMESKIP_V0 = 'FreewayNoFrameskip-v0'

    GRAVITAR_V0 = 'Gravitar-v0'
    GRAVITAR_NO_FRAMESKIP_V0 = 'GravitarNoFrameskip-v0'

    MONTEZUMAS_REVENGE_V0 = 'MontezumaRevenge-v0'
    MONTEZUMAS_REVENGE_NO_FRAMESKIP_V0 = 'MontezumaRevengeNoFrameskip-v0'

    PITFALL_V0 = 'Pitfall-v0'
    PITFALL_NO_FRAMESKIP_V0 = 'PitfallNoFrameskip-v0'

    PRIVATE_EYE_V0 = 'PrivateEye-v0'
    PRIVATE_EYE_NO_FRAMESKIP_V0 = 'PrivateEyeNoFrameskip-v0'

    SOLARIS_V0 = 'Solaris-v0'
    SOLARIS_NO_FRAMESKIP_V0 = 'SolarisNoFrameskip-v0'

    VENTURE_V0 = 'Venture-v0'
    VENTURE_NO_FRAMESKIP_V0 = 'VentureNoFrameskip-v0'
