#!/bin/bash

# Download checkpoints
if [ ! -d "checkpoints" ]; then
    gdown 1aA251WdB4W1gtUUplf4Kda4GsLlrNto9 -O checkpoints.zip
    unzip -n checkpoints.zip
    rm checkpoints.zip
fi
