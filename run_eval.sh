#!/bin/bash
cd /home/huangcy/Graph-Learning-BP/GNN_learning
/home/huangcy/anaconda3/envs/py310/bin/python -u evaluate.py > /tmp/eval_full.log 2>&1
echo "DONE" >> /tmp/eval_full.log
