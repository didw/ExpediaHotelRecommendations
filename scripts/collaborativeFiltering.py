# coding: utf-8
__author__ = 'Ravi: https://kaggle.com/company'

import datetime
from heapq import nlargest
from operator import itemgetter
import math
import numpy as np


userMaxId = 1198785
matR = np.ones((userMaxId+1)*101, dtype='f').reshape(userMaxId+1, 101)

def prepare_arrays_match():
    f = open("../input/data_train.csv", "r")
    f.readline()
    
    total = 0

    # Calc counts
    while 1:
        line = f.readline().strip()
        total += 1

        if total % (3600000*0.8) == 0:
            print('Read {} %'.format(total/(360000*0.8)))

        if line == '':
            break

        arr = line.split(",")
        
        if arr[11] != '':
            book_year = int(arr[11][:4])
            book_month = int(arr[11][5:7])
        else:
            book_year = int(arr[0][:4])
            book_month = int(arr[0][5:7])
            
        if book_month<1 or book_month>12 or book_year<2012 or book_year>2015:
            #print(book_month)
            #print(book_year)
            #print(line)
            continue
            
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        user_id = arr[7]
        is_package = arr[9]
        srch_destination_id = arr[16]
        hotel_country = arr[21]
        hotel_market = arr[22]
        is_booking = float(arr[18])
        hotel_cluster = arr[23]

        append_0 = ((book_year - 2012)*12 + (book_month - 12))
        if not (append_0>0 and append_0<=36):
            #print(book_year)
            #print(book_month)
            #print(line)
            #print(append_0)
            continue
        
        append_1 = pow(math.log(append_0), 1.3) * pow(append_0, 1.46) * (3.5 + 17.6*is_booking)
        append_2 = 3 + 5.56*is_booking


        if user_id != '':
            matR[int(user_id)][int(hotel_cluster)] += 1

    f.close()

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T

def userid_matrix_factorization():
    N = len(matR)
    M = len(matR[0])
    K = 10

    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

    nP, nQ = matrix_factorization(matR, P, Q, K)
    nR = np.dot(nP, nQ.T)
    return nR


def gen_evaluation(nR):
    now = datetime.datetime.now()
    f = open("../input/data_val.csv", "r")
    f.readline()
    total = 0
    score = 0

    while 1:
        line = f.readline().strip()
        total += 1

        if total % (3600000*0.2) == 0:
            print('Evaluation {} %'.format(total/(360000*0.2)))

        if line == '':
            break

        arr = line.split(",")

        #id = arr[0]
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        user_id = arr[7]
        is_package = arr[9]
        srch_destination_id = arr[16]
        hotel_country = arr[21]
        hotel_market = arr[22]
        hotel_cluster = int(arr[23])

        if arr[11] != '':
            ch_month = int(arr[11][5:7])
            book_season = ((ch_month + 9)%12) / 3
        else:
            book_season = -1

        filled = []

        d = nR[user_id]
        topitems = np.argsort(-d)
        for i in range(5):
            if topitems[i] == hotel_cluster:
                score += 1.0 / (i+1)

    print('Result score:{}'.format(score*100.0/total))

prepare_arrays_match()

nR = userid_matrix_factorization()

gen_evaluation(nR)
#gen_submission(best_s00, best_s01,best_hotels_country, best_hotels_search_dest, best_hotels_od_ulc, best_hotels_uid_miss, popular_hotel_cluster)

