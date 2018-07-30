from enum import IntEnum

from ._subproc_env import SubProcEnv, DummyVecEnv


class Engines(IntEnum):
    ATARI = 0
    DOOM = 1
    SC2 = 2


SC2_ENVS = {
    'BuildMarines',
    'CollectMineralShards',
    'DefeatRoaches',
    'DefeatZerglingsAndBanelings',
    'FindAndDefeatZerglings',
    'MoveToBeacon'
}


def reward_normalizer_by_env_id(env_id):
    from adept.utils.normalizers import Clip, Scale
    norm_by_id = {
        'DefeatRoaches': Scale(0.1),
        'DefeatZerglingsAndBanelings': Scale(0.2)
    }
    if env_id not in norm_by_id:
        return Clip()
    else:
        return norm_by_id[env_id]


ATARI_6_ENVS = [
    'BeamRiderNoFrameskip-v4',
    'BreakoutNoFrameskip-v4',
    'QbertNoFrameskip-v4',
    'SeaquestNoFrameskip-v4',
    'SpaceInvadersNoFrameskip-v4',
    'PongNoFrameskip-v4',
]

ATARI_ENVS = [
    # 'AirRaidNoFrameskip-v4',
    # 'AlienNoFrameskip-v4',
    # 'AmidarNoFrameskip-v4',
    # 'AssaultNoFrameskip-v4',
    # 'AsterixNoFrameskip-v4',
    'AsteroidsNoFrameskip-v4',
    # 'AtlantisNoFrameskip-v4',
    # 'BankHeistNoFrameskip-v4',
    # 'BattleZoneNoFrameskip-v4',
    'BeamRiderNoFrameskip-v4',
    # 'BerzerkNoFrameskip-v4',
    # 'BowlingNoFrameskip-v4',
    'BoxingNoFrameskip-v4',
    'BreakoutNoFrameskip-v4',
    # 'CarnivalNoFrameskip-v4',
    'CentipedeNoFrameskip-v4',
    # 'ChopperCommandNoFrameskip-v4',
    # 'CrazyClimberNoFrameskip-v4',
    # 'DemonAttackNoFrameskip-v4',
    # 'DoubleDunkNoFrameskip-v4',
    # 'ElevatorActionNoFrameskip-v4',
    # 'EnduroNoFrameskip-v4',
    # 'FishingDerbyNoFrameskip-v4',
    # 'FreewayNoFrameskip-v4',
    # 'FrostbiteNoFrameskip-v4',
    # 'GopherNoFrameskip-v4',
    # 'GravitarNoFrameskip-v4',
    # 'HeroNoFrameskip-v4',
    # 'IceHockeyNoFrameskip-v4',
    # 'JamesbondNoFrameskip-v4',
    # 'JourneyEscapeNoFrameskip-v4',
    # 'KangarooNoFrameskip-v4',
    # 'KrullNoFrameskip-v4',
    # 'KungFuMasterNoFrameskip-v4',
    # 'MontezumaRevengeNoFrameskip-v4',
    'MsPacmanNoFrameskip-v4',
    # 'NameThisGameNoFrameskip-v4',
    # 'PhoenixNoFrameskip-v4',
    # 'PitfallNoFrameskip-v4',
    # 'PongNoFrameskip-v4',
    # 'PooyanNoFrameskip-v4',
    # 'PrivateEyeNoFrameskip-v4',
    'QbertNoFrameskip-v4',
    # 'RiverraidNoFrameskip-v4',
    # 'RoadRunnerNoFrameskip-v4',
    # 'RobotankNoFrameskip-v4',
    'SeaquestNoFrameskip-v4',
    # 'SkiingNoFrameskip-v4',
    # 'SolarisNoFrameskip-v4',
    'SpaceInvadersNoFrameskip-v4',
    # 'StarGunnerNoFrameskip-v4',
    # 'TennisNoFrameskip-v4',
    # 'TimePilotNoFrameskip-v4',
    # 'TutankhamNoFrameskip-v4',
    # 'UpNDownNoFrameskip-v4',
    # 'VentureNoFrameskip-v4',
    'VideoPinballNoFrameskip-v4',
    # 'WizardOfWorNoFrameskip-v4',
    # 'YarsRevengeNoFrameskip-v4',
    # 'ZaxxonNoFrameskip-v4'
]
