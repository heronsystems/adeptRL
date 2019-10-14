import datetime
import json
import logging
import os

from adept.globals import VERSION
from adept.registry import REGISTRY as R
from adept.utils.script_helpers import LogDirHelper
from adept.utils.util import DotDict


class Init:
    """
    Handles some of the container setup process.

    * initial step count
    * load dynamic args or defaults
    * print logo
    * make logger
    * write args file
    * log args
    * create log dir
    """
    @staticmethod
    def main(mode, args):
        args = DotDict(args)

        if not args.prompt:
            args = Init.from_defaults(args)
        if args.config:
            args = Init.from_config(args)
        if args.prompt:
            args = Init.from_prompt(args)

        if args.agent:
            name = args.agent
        else:
            name = args.actor_host

        log_id = Init.make_log_id(args.tag, mode, name, args.netbody)
        log_id_dir = Init.log_id_dir(args.logdir, args.env, log_id)
        initial_step = 0

        if args.resume:
            args, log_id_dir, initial_step = Init.from_resume(mode, args)

        Init.print_ascii_logo()
        Init.make_log_dirs(log_id_dir)
        Init.write_args_file(log_id_dir, args)
        logger = Init.setup_logger(log_id_dir)
        Init.log_args(logger, args)
        return args, log_id_dir, initial_step, logger

    @staticmethod
    def from_resume(mode, args):
        """
        :param mode: Script name
        :param args: Dict[str, Any], static args
        :return: args, log_id, initial_step_count
        """
        resume = args.resume
        log_dir_helper = LogDirHelper(args.resume)
        with open(log_dir_helper.args_file_path(), 'r') as args_file:
            args = DotDict(json.load(args_file))
            args.resume = resume

        args.load_network = log_dir_helper.latest_network_path()
        args.load_optim = log_dir_helper.latest_optim_path()
        initial_step_count = log_dir_helper.latest_epoch()

        if args.agent:
            name = args.agent
        else:
            name = args.actor_host

        log_id = Init.make_log_id(
            args.tag, mode, name, args.netbody,
            timestamp=log_dir_helper.timestamp()
        )
        log_id_path = Init.log_id_dir(args.logdir, args.env, log_id)
        return args, log_id_path, initial_step_count

    @staticmethod
    def from_defaults(args):
        if args.agent:
            agent_cls = R.lookup_agent(args.agent)
            agent_args = agent_cls.args
        else:
            h = R.lookup_actor(args.actor_host)
            w = R.lookup_actor(args.actor_worker)
            l = R.lookup_learner(args.learner)
            e = R.lookup_exp(args.exp)
            agent_args = {**h.args, **w.args, **l.args, **e.args}

        env_cls = R.lookup_env(args.env)
        rwdnorm_cls = R.lookup_reward_normalizer(args.rwd_norm)

        env_args = env_cls.args
        rwdnorm_args = rwdnorm_cls.args
        if args.custom_network:
            net_args = R.lookup_network(args.custom_network).args
        else:
            net_args = R.lookup_modular_args(args)
        args = DotDict({
            **args, **agent_actor_args, **env_args, **rwdnorm_args, **net_args
        })

        return args

    @staticmethod
    def from_config(args):
        with open(args.config, 'r') as args_file:
            config_args = json.load(args_file)
        args = DotDict({**args, **config_args})
        return args

    @staticmethod
    def from_prompt(args):
        if args.agent:
            agent_cls = R.lookup_agent(args.agent)
            agent_args = agent_cls.prompt(provided=args)
        else:
            h = R.lookup_actor(args.actor_host)
            w = R.lookup_actor(args.actor_worker)
            l = R.lookup_learner(args.learner)
            e = R.lookup_exp(args.exp)
            agent_args = {
                **h.prompt(args), **w.prompt(args),
                **l.prompt(args), **e.prompt(args)
            }

        env_cls = R.lookup_env(args.env)
        rwdnorm_cls = R.lookup_reward_normalizer(args.rwd_norm)

        env_args = env_cls.prompt(provided=args)
        rwdnorm_args = rwdnorm_cls.prompt(provided=args)
        if args.custom_network:
            net_args = R.lookup_network(args.custom_network).prompt()
        else:
            net_args = R.prompt_modular_args(args)
        args = DotDict({
            **args, **agent_args, **env_args, **rwdnorm_args, **net_args
        })
        return args

    @staticmethod
    def print_ascii_logo():
        version_len = len(VERSION)
        print(
            """
                         __           __
              ____ _____/ /__  ____  / /_
             / __ `/ __  / _ \/ __ \/ __/
            / /_/ / /_/ /  __/ /_/ / /_
            \__,_/\__,_/\___/ .___/\__/
                           /_/           """ + '\n' +
            '                                     '[:-(version_len + 2)] +
            'v{} '.format(VERSION)
        )

    @staticmethod
    def make_log_dirs(log_id_dir):
        return os.makedirs(log_id_dir, exist_ok=True)

    @staticmethod
    def setup_logger(log_id_dir, log_name='train'):
        logger = logging.getLogger(log_id_dir + '_' + log_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        fmt = logging.Formatter('%(message)s')
        sh.setFormatter(fmt)

        log_path = os.path.join(log_id_dir, f'{log_name}_log.txt')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter('%(asctime)s  [%(levelname)s] %(message)s')
        fh.setFormatter(fmt)

        logger.addHandler(sh)
        logger.addHandler(fh)

        return logger

    @staticmethod
    def log_args(logger, args):
        args = args if isinstance(args, dict) else vars(args)
        for k in sorted(args):
            logger.info('{}: {}'.format(k, args[k]))

    @staticmethod
    def write_args_file(log_id_dir, args):
        args = args if isinstance(args, dict) else vars(args)
        with open(os.path.join(log_id_dir, 'args.json'), 'w') as args_file:
            json.dump(args, args_file, indent=4, sort_keys=True)

    @staticmethod
    def make_log_id(tag, mode_name, agent_name, network_name, timestamp=None):
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        parts = [mode_name, agent_name, network_name, timestamp]
        if tag:
            parts = [tag] + parts
        return '_'.join(parts)

    @staticmethod
    def log_id_dir(logdir, env, log_id):
        return os.path.join(logdir, env, log_id)
