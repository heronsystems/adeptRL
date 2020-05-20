# Setting up Docker with Adept

1. Install [docker-ce](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
2. Install [CUDA 10](https://developer.nvidia.com/cuda-downloads)
3. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (don't run
 the final line)

It's not required to use the image that corresponds to your OS version since
docker will just download it. However you must make sure the installed CUDA
version matches the CUDA version of the docker image (`FROM
nvidia/cuda:<CUDA_VERSION>-runtime-ubuntu18.04`).

The below assumes that you're running things on an Ubuntu box, though things should sort of work on a Mac as well. Ping Karthik (karthik@starfruit-llc.com) with questions.

# Quickstart
To set things up:

1. Place `Dockerfile`, `connect.py`, and `startup.sh` under a folder in your project directory. E.g., one common way to do this would be to place these files under `your_root_project_directory/docker`.

2. Run the following command, replacing values for `--dockerfile`, `--username`, `--email`, and `--fullname`. In particular, `--username` will refer to your username in the Docker container; feel free to set this to anything you'd like. The script asks for `--email` and `--fullname` so that you can develop code easily within Docker; you'll want to set these values to what you use with GitHub. For example, Karthik's setup would be:

```
python connect.py --dockerfile /path/to/your/Dockerfile --username karthik --email karthik@starfruit-llc.com --fullname "Karthik Narayan"
```

3. This command should fail the first time you run, asking you to create a few directories and symlink your `.ssh` directory. You may get a message similar to the following:

```
Please run: mkdir -p /mnt/users/karthik and run: chmod -R a+rw /mnt/users/karthik. You may need to run these commands using sudo.
Please run the following before running connect.py. This is to allow your Docker instance to be able to pull/push from private github repositories:
ln -s ~/.ssh /mnt/users/karthik/.ssh
```

4. After addressing the failures, simply re-run the `connect.py` command in step (1). This will build the Docker image for you. Go grab your favorite drink - this could take a while.
5. If things succeeded, you'll see a message like the following:
```
Run this to connect:
xhost +"local:docker"; docker run -it --rm --detach-keys="ctrl-@" -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro --network host -v /mnt/:/mnt/ karthik-dev; xhost -"local:docker"
```
6. To create and connect to your Docker container, simply run the above command.

7. To run Adept:
```
cd clients; git clone https://github.com/heronsystems/adeptRL
cd adeptRL
sudo python setup.py develop
python -m adept.scripts.local
```

# Features
Here are a number of default features that the stock setup comes with. The good news is that it can all be very easily modified.

## Persistence
Upon connecting to your Docker instance, you'll notice two folders in your home directory: `clients` and `data`. Any files that you place within these directories are persistent, and will re-appear even if you close out your Docker instance.

You can also access these files outside the Docker container, as they are stored under `/mnt/users/yourusername/clients` and `/mnt/users/yourusername/data`.

Although the usual convention is to place code under `clients` and any data under `data`, you're free to use these folders however you'd like.

## Sudo and Installing Things on the Fly
For most intents and purposes, when you connect to your Docker instance, you've created a new machine. Running `sudo apt-get install`s and `pip install`s will work with no issue; just remember that prior to doing your first `sudo apt-get install`, you'll need to do a `sudo apt-get update`. More generally, `sudo` works (and is password-less).

Feel free to install things to your heart's content - and if things break, no worries; just exit the Docker container, re-connect, and retry. Upon re-connection, all of your `sudo apt-get install`s and `pip install`s will be forgotten.

## Modifying Default Dependencies
To modify the default dependencies, you can simply modify the default Dockerfile. Of course, prior to committing this change, you'll want to check with your team.

## Running GUI Applications
You should be able to run GUI applications within the Docker container, and these should appear.

## Nice Terminal Defaults

There are a number of nice default features equipped with your terminal:

- Features `zsh` rather than `bash`, which has nicer autocomplete functionality.
- Typing `Ctrl-r` will show you a history of commands that you've previously typed into your Docker container. The history should persist across Docker runs, though this doesn't work perfectly.
- By default, you're dropped into a `tmux` session. It's highly recommended that you configure `.tmux.conf`, located under `/mnt/users/yourusername/.tmux.conf`. This will allow you to work with multiple terminals in a split-screen fashion within a single Docker instance.

# Frequently Asked Questions (FAQ)

## Help! I accidentally forgot my command to re-connect to the Docker container.
No worries - go back to `yourproject/docker` and re-type:
```
python connect.py --dockerfile /path/to/your/Dockerfile --username yourusername --email youremail --fullname "yourfullname"
```
Ensure to type in the same values for `--username`, `--email`, and `--fullname` to avoid re-building the Docker container. If you re-type in the same values, this `connect.py` script should finish more-or-less instantly and give you the command to run.

## Is Mac OS supported?
This hasn't been extensively tested on Mac OS, but things should generally work out of the box. When connecting to the docker image, you'll want to get rid of the `xhost` commands and `--network host` setting; so you should connect via something like this:

```docker run -it --rm --detach-keys="ctrl-@" -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v /mnt/:/mnt/ karthik-dev```

Things like GUI applications don't seem to run, but again, likely shouldn't be too difficult to figure out. If you really want things to work on Mac OS, ping Karthik, who can set this up for you.