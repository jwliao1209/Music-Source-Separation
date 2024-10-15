#!/bin/bash

# Download dataset
if [ ! -d "musdb18hq" ]; then
    wget https://zenodo.org/records/3338373/files/musdb18hq.zip
    unzip musdb18hq.zip
    rm musdb18hq.gz
fi
