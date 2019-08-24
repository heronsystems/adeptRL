import json
from datetime import datetime

from adept.utils.script_helpers import LogDirHelper
from adept.utils.util import DotDict


class Local:
    def __init__(
            self,
            agent,
            env,
            rwd_norm,
            device,
            nb_env,
            seed,
            nb_step,
            network,
            optimizer,
            initial_step_count,
            logdir
    ):
        self.initial_step_count = initial_step_count

    @classmethod
    def from_resume(cls, resume_path):
        log_dir_helper = LogDirHelper(resume_path)
        with open(log_dir_helper.args_file_path(), 'r') as args_file:
            args = DotDict(json.load(args_file))

        args.load_network = log_dir_helper.latest_network_path()
        args.load_optim = log_dir_helper.latest_optim_path()
        log_id = cls._make_log_id(
            args.tag, cls.__name__, args.agent, args.netbody,
            timestamp=log_dir_helper.timestamp()
        )
        initial_step_count = log_dir_helper.latest_epoch()
        return cls.from_args(args, log_id, initial_step_count)

    @classmethod
    def from_args(cls, args, log_id=None, initial_step_count=0):
        if log_id is None:
            log_id = cls._make_log_id(
                args.tag, cls.__name__, args.agent, args.netbody
            )





    @staticmethod
    def parse_dynamic_args(args, reg):
        if args.use_defaults:
            agent_args = reg.lookup_agent(args.agent).args
            env_args = reg.lookup_env_class(args.env).args
            rwdnorm_args = reg.lookup_reward_normalizer(args.rwd_norm).args
            if args.custom_network:
                net_args = reg.lookup_network(args.custom_network).args
            else:
                net_args = reg.lookup_modular_args(args)
        else:
            agent_args = reg.lookup_agent(args.agent).prompt()
            env_args = reg.lookup_env(args.env).prompt()
            rwdnorm_args = reg.lookup_reward_normalizer(args.rwd_norm).prompt()
            if args.custom_network:
                net_args = reg.lookup_network(args.custom_network).prompt()
            else:
                net_args = reg.prompt_modular_args(args)
        return DotDict({
            **args, **agent_args, **env_args, **rwdnorm_args, **net_args
        })

    @staticmethod
    def _make_log_id(tag, mode_name, agent_name, network_name, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        if tag:
            log_id = '_'.join([
                tag, mode_name, agent_name, network_name, timestamp
            ])
        else:
            log_id = '_'.join([mode_name, agent_name, network_name, timestamp])
        return log_id