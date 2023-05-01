#!/bin/bash
cd /home/ubuntu
cd style_transfer_app
sanic server:app &
#python3 style_transfer_app/server.py > log.out
