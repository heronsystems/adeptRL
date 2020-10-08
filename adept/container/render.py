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

from adept.network import ModularNetwork
from adept.registry import REGISTRY
from adept.utils import dtensor_to_dev, listd_to_dlist
from adept.utils.script_helpers import LogDirHelper
from adept.utils.util import DotDict


class RenderContainer:
    def __init__(
        self,
        actor,
        epoch_id,
        start,
        end,
        logger,
        log_id_dir,
        gpu_id,
        seed,
        manager,
        extra_args={},
    ):
        self.log_dir_helper = log_dir_helper = LogDirHelper(log_id_dir)
        self.train_args = train_args = log_dir_helper.load_args()
        self.train_args = DotDict(**self.train_args, **extra_args)
        self.device = device = self._device_from_gpu_id(gpu_id)
        self.logger = logger

        if epoch_id:
            epoch_ids = [epoch_id]
        else:
            epoch_ids = self.log_dir_helper.epochs()
            epoch_ids = filter(lambda eid: eid >= start, epoch_ids)
            if end != -1.0:
                epoch_ids = filter(lambda eid: eid <= end, epoch_ids)
            epoch_ids = list(epoch_ids)
        self.epoch_ids = epoch_ids

        engine = REGISTRY.lookup_engine(train_args.env)
        env_cls = REGISTRY.lookup_env(train_args.env)
        manager_cls = REGISTRY.lookup_manager(manager)
        self.env_mgr = manager_cls.from_args(
            self.train_args, engine, env_cls, seed=seed, nb_env=1
        )
        if train_args.agent:
            agent = train_args.agent
        else:
            agent = train_args.actor_host
        output_space = REGISTRY.lookup_output_space(
            agent, self.env_mgr.action_space
        )
        actor_cls = REGISTRY.lookup_actor(actor)
        self.actor = actor_cls.from_args(
            actor_cls.prompt(), self.env_mgr.action_space
        )

        self.network = self._init_network(
            train_args,
            self.env_mgr.observation_space,
            self.env_mgr.gpu_preprocessor,
            output_space,
            REGISTRY,
        ).to(device)

    @staticmethod
    def _device_from_gpu_id(gpu_id):
        return torch.device(
            "cuda:{}".format(gpu_id)
            if (torch.cuda.is_available() and gpu_id >= 0)
            else "cpu"
        )

    @staticmethod
    def _init_network(
        train_args, obs_space, gpu_preprocessor, output_space, net_reg
    ):
        if train_args.custom_network:
            net_cls = net_reg.lookup_network(train_args.custom_network)
        else:
            net_cls = ModularNetwork

        return net_cls.from_args(
            train_args, obs_space, output_space, gpu_preprocessor, net_reg
        )

    def run(self):
        for epoch_id in self.epoch_ids:
            reward_buf = 0
            for net_path in self.log_dir_helper.network_paths_at_epoch(
                epoch_id
            ):
                self.network.load_state_dict(
                    torch.load(
                        net_path, map_location=lambda storage, loc: storage
                    )
                )
                self.network.eval()

                internals = listd_to_dlist(
                    [self.network.new_internals(self.device)]
                )
                next_obs = dtensor_to_dev(self.env_mgr.reset(), self.device)
                self.env_mgr.render()

                episode_complete = False
                while not episode_complete:
                    obs = next_obs
                    with torch.no_grad():
                        actions, _, internals = self.actor.act(
                            self.network, obs, internals
                        )
                    next_obs, rewards, terminals, infos = self.env_mgr.step(
                        actions
                    )
                    self.env_mgr.render()
                    next_obs = dtensor_to_dev(next_obs, self.device)

                    reward_buf += rewards[0]

                    if terminals[0]:
                        episode_complete = True

                print(f"EPOCH_ID: {epoch_id} REWARD: {reward_buf}")

    def close(self):
        self.env_mgr.close()
