# coding: utf-8
__author__ = 'Ravi: https://kaggle.com/company'

import datetime
from heapq import nlargest
from operator import itemgetter
import math
import numpy as np

def prepare_arrays_match():
    f = open("../input/test.csv", "r")
    f.readline()

    best_hotels_od_ulc = dict()
    total = 0

    # Calc counts
    while 1:
        line = f.readline().strip()
        total += 1

        if total % 250000 == 0:
            print('Read {} %'.format(total/25000))

        if line == '':
            break

        arr = line.split(",")

        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        
        if user_location_city != '' and orig_destination_distance != '':
            s1 = (user_location_city, orig_destination_distance)
            best_hotels_od_ulc[s1] = 1
            
    f.close()
    return best_hotels_od_ulc


def gen_submission(best_hotels_od_ulc):
    now = datetime.datetime.now()
    f = open("../input/train.csv", "r")
    out1 = open("../input/train_rmdl.csv", "w")
    out2 = open("../input/train_dl.csv", "w")
    line = f.readline()
    total = 0
    total1 = 0

    out1.write("{}".format(line))
    out2.write("{}".format(line))

    while 1:
        line = f.readline().strip()
        total += 1

        if total % 3600000 == 0:
            print('Write {} %'.format(total/360000))

        if line == '':
            break

        arr = line.split(",")

        user_location_city = arr[5]
        orig_destination_distance = arr[6]

        s1 = (user_location_city, orig_destination_distance)
        if s1 in best_hotels_od_ulc:
            out2.write("{}\n".format(line))
        else:
            out1.write("{}\n".format(line))
            total1 += 1

    f.close()
    out1.close()
    out2.close()
    print('Total  1: {:10d} ...'.format(total1))


best_hotels_od_ulc = prepare_arrays_match()
gen_submission(best_hotels_od_ulc)
