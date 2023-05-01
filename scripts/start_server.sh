#!/bin/bash
cd /home/ubuntu
python3 style_transfer_app/server.py > log.out 2>&1 &
# python3 style_transfer_app/server.py &
# cd style_transfer_app
# sanic server:app &
