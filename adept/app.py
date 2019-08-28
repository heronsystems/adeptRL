#!/usr/bin/env python
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
"""
             __           __
  ____ _____/ /__  ____  / /_
 / __ `/ __  / _ \/ __ \/ __/
/ /_/ / /_/ /  __/ /_/ / /_
\__,_/\__,_/\___/ .___/\__/
               /_/

Usage:
    adept.app <command> [<args>...]
    adept.app (-h | --help)
    adept.app --version

Commands:
    local               Train an agent on a single GPU.
    distrib             Train an agent on multiple machines and/or GPUs.
    actorlearner        Train an agent on multiple machines and/or GPUs.
    evaluate            Evaluate a trained agent.
    render_atari        Visualize an agent playing an Atari game.
    replay_gen_sc2      Generate SC2 replay files of an agent playing SC2.

See 'adept.app help <command>' for more information on a specific command.
"""
from docopt import docopt
from adept.globals import VERSION
from subprocess import call
import os


def parse_args():
    args = docopt(
        __doc__,
        version='adept version ' + VERSION,
        options_first=True
    )

    env = os.environ
    argv = args['<args>']
    if args['<command>'] == 'local':
        exit(call(['python', '-m', 'adept.scripts.local'] + argv, env=env))
    elif args['<command>'] == 'distrib':
        exit(call([
            'python',
            '-m',
            'adept.scripts.distrib'
        ] + argv, env=env))
    elif args['<command>'] == 'actorlearner':
        exit(call([
              'python',
              '-m',
              'adept.scripts.actorlearner'
          ] + argv, env=env))
    elif args['<command>'] == 'evaluate':
        exit(call(['python', '-m', 'adept.scripts.evaluate'] + argv, env=env))
    elif args['<command>'] == 'render_atari':
        exit(call([
            'python', '-m', 'adept.scripts.render_atari'
        ] + argv, env=env))
    elif args['<command>'] == 'replay_gen_sc2':
        exit(call([
            'python', '-m', 'adept.scripts.replay_gen_sc2'
        ] + argv, env=env))
    elif args['<command>'] == 'help':
        if 'local' in args['<args>']:
            exit(call(['python', '-m', 'adept.scripts.local', '-h']))
        elif 'distrib' in args['<args>']:
            exit(call(['python', '-m', 'adept.scripts.distrib', '-h']))
        elif 'actorlearner' in args['<args>']:
            exit(call(['python', '-m', 'adept.scripts.actorlearner', '-h']))
        elif 'evaluate' in args['<args>']:
            exit(call(['python', '-m', 'adept.scripts.evaluate', '-h']))
        elif 'render_atari' in args['<args>']:
            exit(call(['python', '-m', 'adept.scripts.render_atari', '-h']))
        elif 'replay_gen_sc2' in args['<args>']:
            exit(call(['python', '-m', 'adept.scripts.replay_gen_sc2', '-h']))
    else:
        exit(
            "{} is not a valid command. See 'adept.app --help'.".
                format(args['<command>'])
        )


if __name__ == '__main__':
    parse_args()
