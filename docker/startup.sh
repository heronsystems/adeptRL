#!/usr/bin/zsh
# Copyright (C) 2020 Heron Systems, Inc.
#
# This script's purpose is to run preliminary setup for the dev environment
# once an image has been created.

# retain state across sessions
mkdir -p "/mnt/users/$USERNAME"
sudo chown -R $USERNAME "/mnt/users/$USERNAME"
cd /home/$USERNAME/
persistents=".bashrc .bash_logout .zshrc"
for f in ${persistents};
    do if [ ! -e /mnt/users/$USERNAME/${f} ];
    then
        cp ${f} /mnt/users/$USERNAME/${f};
    fi;
done

# zhistory and clients aren't created by default, so create them if they don't
# exist already
test -e /mnt/users/$USERNAME/.zhistory || touch /mnt/users/$USERNAME/.zhistory
mkdir -p /mnt/users/$USERNAME/clients
mkdir -p /mnt/users/$USERNAME/data

# symlink these files to the persistent versions
persistents=".bashrc .bash_logout .zshrc .zhistory .tmux.conf .ssh clients data"
for f in ${persistents};
    do rm -f ${f} && ln -s "/mnt/users/${USERNAME}/$f" $f;
done

# git setup
export GIT_SSL_NO_VERIFY=1
git config --global user.email "$EMAIL"
git config --global user.name "$FULLNAME"
