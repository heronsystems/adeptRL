# Copyright (C) 2020 Heron Systems, Inc.
#
# Sample command:
# python connect.py --dockerfile ./Dockerfile --username gburdell \
#                   --email gburdell@heronsystems.com \
#                   --fullname "George P. Burdell"
#
# Given a path to a Dockerfile, this script (1) builds a Docker image from the
# Dockerfile and (2) outputs a command that can directly be pasted into a shell
# that connects to a Docker instance spawned from the Docker image built in (1).
import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dockerfile",
        default="./Dockerfile",
        help=(
            "Path to the Dockerfile you wish to work with, "
            "e.g., /some/path/to/Dockerfile"
        ),
    )
    parser.add_argument(
        "--username",
        required=True,
        help="Your local username in the docker instance that you connect to.",
    )
    parser.add_argument(
        "--email",
        required=True,
        help=(
            "Your email address, used for git purposes, e.g., gburdell@heronsystems.com."
        ),
    )
    parser.add_argument(
        "--fullname",
        required=True,
        help="Your full name, used for git purposes.",
    )
    return parser.parse_args()


def runcmd(cmd):
    print("Running --\n{}\n--".format(cmd))
    return os.system(cmd)


def connect_local(args):
    # Check for .ssh under /mnt directory; this is important to be able to pull
    # from github within Docker instances.
    if not os.path.exists(
        "/mnt/users/{username}".format(username=args.username)
    ):
        print(
            (
                "Please create the directory /mnt/users/{username} and run:"
                "chmod -R a+rw /mnt/users/{username}."
            ).format(username=args.username)
        )

    # Check for .ssh under /mnt directory; this is important to be able to pull
    # from github within Docker instances.
    if not os.path.exists(
        "/mnt/users/{username}/.ssh".format(username=args.username)
    ):
        print(
            "Please run the following before running connect.py. This is to "
            "allow your Docker instance to be able to pull/push from "
            "private github repositories:\n"
            "ln -s /mnt/users/{username}/.ssh ~/.ssh".format(
                username=args.username
            )
        )
        sys.exit(1)

    # Build the docker image
    dockerfile_dir = os.path.dirname(os.path.abspath(args.dockerfile))
    cmd = (
        "cd {dockerfile_dir}; nvidia-docker build "
        "-f {dockerfile} "
        "--build-arg USERNAME={username} "
        "--build-arg EMAIL={email} "
        '--build-arg FULLNAME="{fullname}" '
        "-t {username}-dev .".format(
            dockerfile=args.dockerfile,
            dockerfile_dir=dockerfile_dir,
            username=args.username,
            email=args.email,
            fullname=args.fullname,
        )
    )
    success = runcmd(cmd)

    if success != 0:
        print("Please fix the errors that occurred above.")
        sys.exit(1)

    # Construct the docker instance creation command
    cmd = (
        'xhost +"local:docker"; nvidia-docker run -it --rm '
        # Take care of ctrl-p issues
        '--detach-keys="ctrl-@" '
        # Configure X
        "-e DISPLAY=$DISPLAY "
        "-v /tmp/.X11-unix:/tmp/.X11-unix:ro "
        # Expose all ports
        "--network host "
        # Mount volumes
        "-v /mnt/:/mnt/ "
        '{}-dev; xhost -"local:docker"'.format(args.username)
    )
    print("Run this to connect:\n{cmd}".format(cmd=cmd))

    # For convenience, copy to the clipboard if possible
    try:
        import pyperclip

        pyperclip.copy(cmd)
    except:
        pass


def main():
    connect_local(parse_args())


if __name__ == "__main__":
    main()
