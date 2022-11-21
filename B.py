# -*- coding: utf-8 -*-
# @author: Kevin
# @email: 2112164129@e.gzhu.edu.cn
# @date: 2022/11/18

import argparse
import numpy as np

parser = argparse.ArgumentParser('测试')
parser.add_argument('param1', type=str, help='姓')
parser.add_argument('param2', type=str, help='名')
args = parser.parse_args()

if __name__ == "__main__":
    print(args.param1 + args.param2)
