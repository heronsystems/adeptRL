#!/bin/bash

sudo docker run \
    --rm \
    -u root \
    -it \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$HOME":/home \
    -p 8080:8080 \
    jenkinsci/blueocean
