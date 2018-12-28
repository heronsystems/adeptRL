#!/usr/env/python

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

if __name__ == '__main__':

    args = docopt(__doc__, version='adept version ' + VERSION)

    print('hello')
    print('global args')
    print(args)
    argv = [args['<command>']] + args['<args>']
    if args['<command>'] == 'local':
        print('calling local')
    elif args['<command>'] == 'towered':
        print('calling towered')
    else:
        exit(
            "{} is not a valid command. See 'adept.app --help'.".
                format(args['<command>'])
        )
