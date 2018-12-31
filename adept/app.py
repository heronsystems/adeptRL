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
    towered             Train an agent on multiple GPUs.
    impala              Train an agent on multiple GPUs with IMPALA.
    evaluate            Evaluate a trained agent.
    resume_local        Resume a local job.
    render_atari        Visualize an agent playing an Atari game.
    replay_gen_sc2      Generate SC2 replay files of an agent playing SC2.

See 'adept.app <command> --help' for more information on a specific command.
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

    print(args)

    env = os.environ
    argv = args['<args>']
    if args['<command>'] == 'local':
        print(argv)
        exit(call(['python', '-m', 'adept.scripts.local'] + argv, env=env))
    elif args['<command>'] == 'towered':
        print('calling towered')
    elif args['<command>'] == 'help':
        if 'local' in args['<args>']:
            exit(call(['python', '-m', 'adept.scripts.local', '-h']))
        # TODO help messages for other commands
    else:
        exit(
            "{} is not a valid command. See 'adept.app --help'.".
                format(args['<command>'])
        )


if __name__ == '__main__':
    parse_args()
