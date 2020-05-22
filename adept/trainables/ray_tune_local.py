import ray
from ray import tune
from ray.tune import track
from time import time
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from adept.network import ModularNetwork
from adept.registry import REGISTRY
from adept.utils.logging import SimpleModelSaver
from adept.utils.util import dtensor_to_dev, listd_to_dlist
from adept.container.base import Container
from adept.container.local import Local
import os
from adept.utils.util import DotDict


class Trainable(tune.Trainable):

    def _setup(self,  config):
        logger = config['logger']
        log_id_dir = config['log_id_dir']
        initial_step = config['initial_step']
        config.gpu_id = torch.cuda.current_device()
        self.local = Local(config, logger, log_id_dir, initial_step)

    def _train(self):
        """Run 1 step of training (e.g., one epoch).

        Returns:
            A dict of training metrics.
        """
        term_reward = float(self.local.run())
        return {'term_reward': term_reward}

    def _save(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.local.network.state_dict(), checkpoint_path)
        print('SAVING MODEL ---------------------', checkpoint_path)
        return checkpoint_path

    def _restore(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.local.network.load_state_dict(torch.load(checkpoint_path))

    def reset_config(self, config):
        for param_group in self.optimizer.param_groups:
            if "lr" in config:
                param_group["lr"] = config["lr"]
                print("REUSING ACTOR IS UPDATING LR----\n\n\---\n\n\n--\n". config['r'])



        self.model = ConvNet()
        self.config = new_config
        return True

    # def _save(self, checkpoint_dir):
    #     self.local.saver.save_state_dicts(
    #             self.local.network, self.local.step_count, self.local.optimizer
    #         )
    #     self.model_dir = os.path.join(self._log_id_dir,
    #                                   str(self.local.step_count),
    #                                   'model_{}.pth'.format(self.local.step_count))
    #
    # def _restore(self, checkpoint_path):
    #     self.model_dir
    #     self.local.network = self.local.load_network(self.local.network, os.path.join(self.args.logdir, str(self.local.step_count)))
    #     self.local.logger.info('Loaded network from {}'.format(self.args.logdir))


class Local(Local):
    def __init__(self, args, logger, log_id_dir, initial_step_count):
        super().__init__(args, logger, log_id_dir, initial_step_count)
        self.step_count = self.initial_step_count

    def run(self):
        epochs_passed = 0
        # next_save = self.init_next_save(self.initial_step_count, self.epoch_len)
        prev_step_t = time()
        ep_rewards = torch.zeros(self.nb_env)

        obs = dtensor_to_dev(self.env_mgr.reset(), self.device)
        internals = listd_to_dlist([
            self.network.new_internals(self.device) for _ in
            range(self.nb_env)
        ])
        step_count = 0
        start_time = time()
        term_rewards_list = []
        while step_count < self.nb_step:
            actions, internals = self.agent.act(self.network, obs, internals)
            next_obs, rewards, terminals, infos = self.env_mgr.step(actions)
            next_obs = dtensor_to_dev(next_obs, self.device)

            self.agent.observe(
                obs,
                rewards.to(self.device).float(),
                terminals.to(self.device).float(),
                infos
            )

            # Perform state updates
            self.step_count += self.nb_env
            step_count += self.nb_env
            ep_rewards += rewards.float()
            obs = next_obs

            term_rewards, term_infos = [], []
            for i, terminal in enumerate(terminals):
                if terminal:
                    for k, v in self.network.new_internals(self.device).items():
                        internals[k][i] = v
                    term_rewards.append(ep_rewards[i].item())
                    if infos[i]:
                        term_infos.append(infos[i])
                    ep_rewards[i].zero_()

            if term_rewards:
                term_reward = np.mean(term_rewards)
                term_rewards_list.append(term_reward.item())
                delta_t = time() - start_time

                if term_infos:
                    float_keys = [
                        k for k, v in term_infos[0].items() if type(v) == float
                    ]
                    term_infos_dlist = listd_to_dlist(term_infos)
                    for k in float_keys:
                        self.summary_writer.add_scalar(
                            f'info/{k}',
                            np.mean(term_infos_dlist[k]),
                            self.step_count
                        )

            # Learn
            if self.agent.is_ready():
                loss_dict, total_loss, metric_dict = self.agent.compute_loss_and_step(
                    self.network, self.optimizer, next_obs, internals
                )
                epoch = self.step_count / self.nb_env
                self.scheduler.step(epoch)

                self.agent.clear()
                for k, vs in internals.items():
                    internals[k] = [v.detach() for v in vs]

                # write summaries
                cur_step_t = time()
                if cur_step_t - prev_step_t > self.summary_freq:
                    self.write_summaries(
                        self.summary_writer, self.step_count, total_loss, loss_dict,
                        metric_dict, self.network.named_parameters()
                    )
                    prev_step_t = cur_step_t
        return np.mean(term_rewards_list)

