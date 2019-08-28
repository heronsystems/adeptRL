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
import torch

from adept.manager import SubProcEnvManager
from adept.network import ModularNetwork
from adept.registry import REGISTRY
from .base import Container


class ActorLearnerHost(Container):
    def __init__(
            self,
            args,
            logger,
            log_id_dir,
            initial_step_count,
            local_rank,
            global_rank,
            world_size
    ):
        seed = args.seed \
            if global_rank == 0 \
            else args.seed + args.nb_env * global_rank
        logger.info('Using {} for rank {} seed.'.format(seed, global_rank))
        # ENV
        env_cls = REGISTRY.lookup_env(args.env)
        env = env_cls.from_args(args.env, 0)
        env.close()

        # NETWORK
        torch.manual_seed(args.seed)
        device = torch.device("cuda:{}".format(local_rank))
        output_space = REGISTRY.lookup_output_space(
            args.agent, env.action_space)
        if args.custom_network:
            net_cls = REGISTRY.lookup_network(args.custom_network)
        else:
            net_cls = ModularNetwork
        net = net_cls.from_args(
            args,
            env.observation_space,
            output_space,
            env.gpu_preprocessor,
            REGISTRY
        )
        logger.info('Network parameters: ' + str(self.count_parameters(net)))

        def optim_fn(x):
            return torch.optim.RMSprop(x, lr=args.lr, eps=1e-5, alpha=0.99)

        # AGENT
        rwd_norm = REGISTRY.lookup_reward_normalizer(
            args.rwd_norm).from_args(args)
        learner_cls = REGISTRY.lookup_learner(args.learner)



class ActorLearnerWorker(Container):
    """
    Actor Learner Architecture worker.
    """

    def __init__(
        self,
        args,
        global_rank,
        world_size,
        device
    ):
        self._agent = agent
        self._environment = environment
        self._nb_env = nb_env
        self._logger = logger

    def run(self, nb_step, initial_step_count=None):
        pass
