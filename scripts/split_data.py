# coding: utf-8
__author__ = 'Ravi: https://kaggle.com/company'

import datetime
from heapq import nlargest
from operator import itemgetter
import math
import numpy as np

def gen_submission():
    now = datetime.datetime.now()
    f = open("../input/train.csv", "r")
    out_train = open("../input/data_train.csv", "w")
    out_val = open("../input/data_val.csv", "w")
    line = f.readline()
    total = 0
    total1 = 0

    out_train.write("{}".format(line))
    out_val.write("{}".format(line))

    while 1:
        line = f.readline().strip()
        total += 1

        if total % 3600000 == 0:
            print('Write {} %'.format(total/360000))

        if line == '':
            break

        r = np.random.uniform(1,100)
        if r <= 20:
            out_val.write("{}\n".format(line))
        else:
            out_train.write("{}\n".format(line))

    f.close()
    out_train.close()
    out_val.close()
    print('Total  1: {:10d} ...'.format(total1))


gen_submission()
