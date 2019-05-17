#!/usr/bin/env bash

sudo apt-get update
sudo apt-get install python-dev graphviz libgraphviz-dev pkg-config
sudo pip install -r requirements.txt
sudo pip install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"
