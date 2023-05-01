#!/bin/bash
cd /home/ubuntu
pwd
ls
nohup python3 style_transfer_app/server.py > log.out
