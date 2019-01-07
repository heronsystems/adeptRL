# Copyright (C) 2018 Heron Systems, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from enum import IntEnum
from adept.environments.env_module import EnvModule
from adept.utils.normalizers import Clip, Scale
from collections import defaultdict


class Engines(IntEnum):
    GYM = 0
    DOOM = 1
    SC2 = 2


SC2_ENVS = [
    'BuildMarines', 'CollectMineralShards', 'DefeatRoaches',
    'DefeatZerglingsAndBanelings', 'FindAndDefeatZerglings', 'MoveToBeacon'
]
ATARI_6_ENVS = [
    'BeamRiderNoFrameskip-v4',
    'BreakoutNoFrameskip-v4',
    'QbertNoFrameskip-v4',
    'SeaquestNoFrameskip-v4',
    'SpaceInvadersNoFrameskip-v4',
    'PongNoFrameskip-v4',
]
ATARI_ENVS = [
    'AirRaidNoFrameskip-v4', 'AlienNoFrameskip-v4', 'AmidarNoFrameskip-v4',
    'AssaultNoFrameskip-v4', 'AsterixNoFrameskip-v4', 'AsteroidsNoFrameskip-v4',
    'AtlantisNoFrameskip-v4', 'BankHeistNoFrameskip-v4',
    'BattleZoneNoFrameskip-v4', 'BeamRiderNoFrameskip-v4',
    'BerzerkNoFrameskip-v4', 'BowlingNoFrameskip-v4', 'BoxingNoFrameskip-v4',
    'BreakoutNoFrameskip-v4', 'CarnivalNoFrameskip-v4',
    'CentipedeNoFrameskip-v4', 'ChopperCommandNoFrameskip-v4',
    'CrazyClimberNoFrameskip-v4', 'DemonAttackNoFrameskip-v4',
    'DoubleDunkNoFrameskip-v4', 'ElevatorActionNoFrameskip-v4',
    'EnduroNoFrameskip-v4', 'FishingDerbyNoFrameskip-v4',
    'FreewayNoFrameskip-v4', 'FrostbiteNoFrameskip-v4', 'GopherNoFrameskip-v4',
    'GravitarNoFrameskip-v4', 'HeroNoFrameskip-v4', 'IceHockeyNoFrameskip-v4',
    'JamesbondNoFrameskip-v4', 'JourneyEscapeNoFrameskip-v4',
    'KangarooNoFrameskip-v4', 'KrullNoFrameskip-v4',
    'KungFuMasterNoFrameskip-v4', 'MontezumaRevengeNoFrameskip-v4',
    'MsPacmanNoFrameskip-v4', 'NameThisGameNoFrameskip-v4',
    'PhoenixNoFrameskip-v4', 'PitfallNoFrameskip-v4', 'PongNoFrameskip-v4',
    'PooyanNoFrameskip-v4', 'PrivateEyeNoFrameskip-v4', 'QbertNoFrameskip-v4',
    'RiverraidNoFrameskip-v4', 'RoadRunnerNoFrameskip-v4',
    'RobotankNoFrameskip-v4', 'SeaquestNoFrameskip-v4', 'SkiingNoFrameskip-v4',
    'SolarisNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4',
    'StarGunnerNoFrameskip-v4', 'TennisNoFrameskip-v4',
    'TimePilotNoFrameskip-v4', 'TutankhamNoFrameskip-v4',
    'UpNDownNoFrameskip-v4', 'VentureNoFrameskip-v4',
    'VideoPinballNoFrameskip-v4', 'WizardOfWorNoFrameskip-v4',
    'YarsRevengeNoFrameskip-v4', 'ZaxxonNoFrameskip-v4'
]


class EnvModuleRegistry:
    """
    Keeps track of supported environment modules.
    """

    def __init__(self):
        self._engine_ids_by_env_id_set = {}
        self._module_class_by_engine_id = {}
        self._reward_norm_by_env_id = defaultdict(lambda: Clip())

        from adept.environments.openai_gym import AdeptGymEnv
        self.register_env(Engines.GYM, AdeptGymEnv, ATARI_ENVS)
        try:
            from adept.environments.deepmind_sc2 import AdeptSC2Env
            self.register_env(Engines.SC2, AdeptSC2Env, SC2_ENVS)
            self.register_reward_normalizer('DefeatRoaches', Scale(0.1))
            self.register_reward_normalizer(
                'DefeatZerglingsAndBanelings', Scale(0.2)
            )
        except ImportError:
            print('StarCraft 2 Environment not detected.')

    def register_env(self, engine_id, env_module_class, env_id_set):
        # TODO assert no duplicate env_ids
        assert issubclass(env_module_class, EnvModule)
        env_module_class.check_defaults()
        self._engine_ids_by_env_id_set[frozenset(env_id_set)] = engine_id
        self._module_class_by_engine_id[engine_id] = env_module_class

    def lookup_env_class(self, env_id):
        engine = self.lookup_engine(env_id)
        return self._module_class_by_engine_id[engine]

    def lookup_engine(self, env_id):
        eng = None
        for env_id_set, engine_id in self._engine_ids_by_env_id_set.items():
            if env_id in env_id_set:
                eng = engine_id
        if eng is None:
            raise Exception('Environment not registered: ' + env_id)
        return eng

    def register_reward_normalizer(self, env_id, normalizer):
        """
        Associate a reward normalizer with an environment id.

        :param env_id: str
        :param normalizer: Callable[[float], float]
        :return:
        """
        self._reward_norm_by_env_id[env_id] = normalizer

    def lookup_reward_normalizer(self, env_id):
        """

        :param env_id: str
        :return: Callable[[float], float]
        """
        return self._reward_norm_by_env_id[env_id]
