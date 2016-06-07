# coding: utf-8
__author__ = 'Ravi: https://kaggle.com/company'

import datetime
from heapq import nlargest
from operator import itemgetter
import math
import numpy as np


userMaxId = 1198785
destMaxId = 65107
destList = np.zeros(destMaxId+1)
# Recommend based on user id 1,198,785 x 100 = 119,878,500 (~500MB)
# Recommend based on dest id 65,107 x 100 = 6,510,700 (~25MB)
cl_user_b = np.ones((userMaxId+1)*101, dtype='f').reshape(userMaxId+1, 101)
cl_dest_b = np.ones((destMaxId+1)*101, dtype='f').reshape(destMaxId+1, 101)
#cl_user_c = np.ones((userMaxId+1)*101, dtype='f').reshape(userMaxId+1, 101)
#cl_dest_c = np.ones((destMaxId+1)*101, dtype='f').reshape(destMaxId+1, 101)
cl_user = np.ones((userMaxId+1)*101, dtype='f').reshape(userMaxId+1, 101)
cl_dest = np.ones((destMaxId+1)*101, dtype='f').reshape(destMaxId+1, 101)

def prepare_arrays_match():
    f = open("../input/data_train.csv", "r")
    f.readline()
    
    best_hotels_od_ulc = dict()
    best_hotels_uid_miss = dict()
    best_hotels_search_dest = dict()
    best_hotels_country = dict()
    popular_hotel_cluster = dict()
    best_s00 = dict()
    best_s01 = dict()
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

        if arr[11] != '':
            ch_month = int(arr[11][5:7])
            book_season = ((ch_month + 9)%12) / 3
        else:
            book_season = -1

        append_0 = ((book_year - 2012)*12 + (book_month - 12))
        if not (append_0>0 and append_0<=36):
            #print(book_year)
            #print(book_month)
            #print(line)
            #print(append_0)
            continue
        
        append_1 = pow(math.log(append_0), 1.3) * pow(append_0, 1.46) * (3.5 + 17.6*is_booking)
        append_2 = 3 + 5.56*is_booking

        if user_location_city != '' and orig_destination_distance != '' and user_id !='' and srch_destination_id != '' and hotel_country != '' and is_booking==1:
            s00 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
            if s00 in best_s00:
                if hotel_cluster in best_s00[s00]:
                    best_s00[s00][hotel_cluster] += append_0
                else:
                    best_s00[s00][hotel_cluster] = append_0
            else:
                best_s00[s00] = dict()
                best_s00[s00][hotel_cluster] = append_0

        if user_location_city != '' and orig_destination_distance != '' and user_id !='' and srch_destination_id != '' and is_booking==1:
            s01 = (user_id, srch_destination_id, hotel_country, hotel_market)
            if s01 in best_s01:
                if hotel_cluster in best_s01[s01]:
                    best_s01[s01][hotel_cluster] += append_0
                else:
                    best_s01[s01][hotel_cluster] = append_0
            else:
                best_s01[s01] = dict()
                best_s01[s01][hotel_cluster] = append_0


        if user_location_city != '' and orig_destination_distance == '' and user_id !='' and srch_destination_id != '' and hotel_country != '' and is_booking==1:
            s0 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
            if s0 in best_hotels_uid_miss:
                if hotel_cluster in best_hotels_uid_miss[s0]:
                    best_hotels_uid_miss[s0][hotel_cluster] += append_0
                else:
                    best_hotels_uid_miss[s0][hotel_cluster] = append_0
            else:
                best_hotels_uid_miss[s0] = dict()
                best_hotels_uid_miss[s0][hotel_cluster] = append_0

        if user_location_city != '' and orig_destination_distance != '':
            s1 = (user_location_city, orig_destination_distance)

            if s1 in best_hotels_od_ulc:
                if hotel_cluster in best_hotels_od_ulc[s1]:
                    best_hotels_od_ulc[s1][hotel_cluster] += append_0
                else:
                    best_hotels_od_ulc[s1][hotel_cluster] = append_0
            else:
                best_hotels_od_ulc[s1] = dict()
                best_hotels_od_ulc[s1][hotel_cluster] = append_0

        if srch_destination_id != '' and hotel_country != '' and hotel_market != '' and is_package != '':
            s2 = (srch_destination_id,hotel_country,hotel_market,is_package)
            if s2 in best_hotels_search_dest:
                if hotel_cluster in best_hotels_search_dest[s2]:
                    best_hotels_search_dest[s2][hotel_cluster] += append_1
                else:
                    best_hotels_search_dest[s2][hotel_cluster] = append_1
            else:
                best_hotels_search_dest[s2] = dict()
                best_hotels_search_dest[s2][hotel_cluster] = append_1

        if hotel_market != '':
            s3 = (hotel_market)
            if s3 in best_hotels_country:
                if hotel_cluster in best_hotels_country[s3]:
                    best_hotels_country[s3][hotel_cluster] += append_2
                else:
                    best_hotels_country[s3][hotel_cluster] = append_2
            else:
                best_hotels_country[s3] = dict()
                best_hotels_country[s3][hotel_cluster] = append_2

        if hotel_cluster in popular_hotel_cluster:
            popular_hotel_cluster[hotel_cluster] += append_0
        else:
            popular_hotel_cluster[hotel_cluster] = append_0


        if user_id != '' and srch_destination_id != '':
            destList[int(srch_destination_id)] = 1
            cl_user_b[int(user_id)][int(hotel_cluster)] += 1
            cl_dest_b[int(srch_destination_id)][int(hotel_cluster)] += 1

    f.close()
    return best_s00,best_s01, best_hotels_country, best_hotels_od_ulc, best_hotels_uid_miss, best_hotels_search_dest, popular_hotel_cluster


def gen_evaluation(best_s00, best_s01,best_hotels_country, best_hotels_search_dest, best_hotels_od_ulc, best_hotels_uid_miss, popular_hotel_cluster):
    now = datetime.datetime.now()
    f = open("../input/data_val.csv", "r")
    f.readline()
    total = 0
    total0 = 0
    total00 = 0
    total1 = 0
    total12 = 0
    total2 = 0
    total3 = 0
    total4 = 0
    w1 = 3.0 # user_id
    w2 = 3.0 # dest_id
    w3 = 2.0 # best_hotels_search_dest
    w4 = 1.0 # best_hotels_country
    w5 = 1.0 # topclasters
    ## (1,2,3,4,5) = evaluation => real
    ## (1,1,1,1,1) = 52.71486   => 0.43764
    ## (2,2,2,1,1) = 54.35500
    ## (3,3,2,1,1) = 54.59103

    topclasters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

    score = 0.0

    #w = 0
    #cl_user = cl_user_b / cl_user_b.sum(axis=1).reshape(userMaxId+1, 1) + w*cl_user_c / cl_user_c.sum(axis=1).reshape(userMaxId+1, 1)
    #cl_dest = cl_dest_b / cl_dest_b.sum(axis=1).reshape(destMaxId+1, 1) + w*cl_dest_c / cl_dest_c.sum(axis=1).reshape(destMaxId+1, 1)
    cl_user = cl_user_b / cl_user_b.sum(axis=1).reshape(userMaxId+1, 1)
    cl_dest = cl_dest_b / cl_dest_b.sum(axis=1).reshape(destMaxId+1, 1)

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
        hotel_cluster = arr[23]

        if arr[11] != '':
            ch_month = int(arr[11][5:7])
            book_season = ((ch_month + 9)%12) / 3
        else:
            book_season = -1


        filled = []

        s1 = (user_location_city, orig_destination_distance)
        if s1 in best_hotels_od_ulc:
            d = best_hotels_od_ulc[s1]
            topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if int(topitems[i][0]) in filled:
                    continue
                if len(filled) == 5:
                    break
                filled.append(int(topitems[i][0]))
                total1 += 1

        if orig_destination_distance == '':
            s0 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
            if s0 in best_hotels_uid_miss:
                d = best_hotels_uid_miss[s0]
                topitems = nlargest(4, sorted(d.items()), key=itemgetter(1))
                for i in range(len(topitems)):
                    if int(topitems[i][0]) in filled:
                        continue
                    if len(filled) == 5:
                        break
                    filled.append(int(topitems[i][0]))
                    total0 += 1

        s00 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
        s01 = (user_id, srch_destination_id, hotel_country, hotel_market)
        if s01 in best_s01 and s00 not in best_s00:
            d = best_s01[s01]
            topitems = nlargest(4, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if int(topitems[i][0]) in filled:
                    continue
                if len(filled) == 5:
                    break
                filled.append(int(topitems[i][0]))
                total00 += 1

        res = np.zeros(101, dtype='f')
        if not(int(srch_destination_id) > destMaxId or srch_destination_id == '' or destList[int(srch_destination_id)] == 0):
            d1 = np.array(cl_user[int(user_id)])
            d2 = np.array(cl_dest[int(srch_destination_id)])
            res += w1 * d1
            res += d2 * d1

        s2 = (srch_destination_id,hotel_country,hotel_market,is_package)
        if s2 in best_hotels_search_dest:
            d = best_hotels_search_dest[s2]
            dsum = 0.01
            for k,v in d.items():
                dsum += int(v)
            for k,v in d.items():
                res[int(k)] += w3 * int(v)/dsum

        s3 = (hotel_market)
        if s3 in best_hotels_country:
            d = best_hotels_country[s3]
            dsum = 0.01
            for k,v in d.items():
                dsum += int(v)
            for k,v in d.items():
                res[int(k)] += w4 * int(v)/dsum

        dsum = 0.01
        for i in range(len(topclasters)):
            dsum += int(topclasters[i][0])
        for i in range(len(topclasters)):
            res[i] += w5 * int(topclasters[i][0])/dsum

        topitems = np.argsort(-res)
        for i in range(5):
            if int(topitems[i]) in filled:
                continue
            if len(filled) == 5:
                break
            filled.append(int(topitems[i]))
            total2 += 1
        for i, v in enumerate(filled):
            if v == int(hotel_cluster):
                score += 1/(i+1)
    print('Result score:{:.5f}'.format(score*100.0/total))

def gen_submission(best_s00, best_s01,best_hotels_country, best_hotels_search_dest, best_hotels_od_ulc, best_hotels_uid_miss, popular_hotel_cluster):
    now = datetime.datetime.now()
    path = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    out = open(path, "w")
    f = open("../input/test.csv", "r")
    f.readline()
    total = 0
    total0 = 0
    total00 = 0
    total1 = 0
    total12 = 0
    total2 = 0
    total3 = 0
    total4 = 0
    w1 = 3.0 # user_id
    w2 = 3.0 # dest_id
    w3 = 2.0 # best_hotels_search_dest
    w4 = 1.0 # best_hotels_country
    w5 = 1.0 # topclasters
    out.write("id,hotel_cluster\n")
    topclasters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))


    #w = 0
    #cl_user = cl_user_b / cl_user_b.sum(axis=1).reshape(userMaxId+1, 1) + w*cl_user_c / cl_user_c.sum(axis=1).reshape(userMaxId+1, 1)
    #cl_dest = cl_dest_b / cl_dest_b.sum(axis=1).reshape(destMaxId+1, 1) + w*cl_dest_c / cl_dest_c.sum(axis=1).reshape(destMaxId+1, 1)
    cl_user = cl_user_b / cl_user_b.sum(axis=1).reshape(userMaxId+1, 1)
    cl_dest = cl_dest_b / cl_dest_b.sum(axis=1).reshape(destMaxId+1, 1)

    while 1:
        line = f.readline().strip()
        total += 1

        if total % 250000 == 0:
            print('Write {} %'.format(total/25000))

        if line == '':
            break

        arr = line.split(",")

        id = arr[0]
        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        user_id = arr[8]
        is_package = arr[10]
        srch_destination_id = arr[17]
        hotel_country = arr[20]
        hotel_market = arr[21]

        if arr[12] != '':
            ch_month = int(arr[12][5:7])
            book_season = ((ch_month + 9)%12) / 3
        else:
            book_season = -1


        out.write(str(id) + ',')
        filled = []

        s1 = (user_location_city, orig_destination_distance)
        if s1 in best_hotels_od_ulc:
            d = best_hotels_od_ulc[s1]
            topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if int(topitems[i][0]) in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(int(topitems[i][0]))
                total1 += 1

        if orig_destination_distance == '':
            s0 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
            if s0 in best_hotels_uid_miss:
                d = best_hotels_uid_miss[s0]
                topitems = nlargest(4, sorted(d.items()), key=itemgetter(1))
                for i in range(len(topitems)):
                    if int(topitems[i][0]) in filled:
                        continue
                    if len(filled) == 5:
                        break
                    out.write(' ' + topitems[i][0])
                    filled.append(int(topitems[i][0]))
                    total0 += 1

        s00 = (user_id, user_location_city, srch_destination_id, hotel_country, hotel_market)
        s01 = (user_id, srch_destination_id, hotel_country, hotel_market)
        if s01 in best_s01 and s00 not in best_s00:
            d = best_s01[s01]
            topitems = nlargest(4, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if int(topitems[i][0]) in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(int(topitems[i][0]))
                total00 += 1

        res = np.zeros(101, dtype='f')
        if not(int(srch_destination_id) > destMaxId or srch_destination_id == '' or destList[int(srch_destination_id)] == 0):
            d1 = np.array(cl_user[int(user_id)])
            d2 = np.array(cl_dest[int(srch_destination_id)])
            res += w1 * d1
            res += d2 * d1

        s2 = (srch_destination_id,hotel_country,hotel_market,is_package)
        if s2 in best_hotels_search_dest:
            d = best_hotels_search_dest[s2]
            dsum = 0.01
            for k,v in d.items():
                dsum += int(v)
            for k,v in d.items():
                res[int(k)] += w3 * int(v)/dsum

        s3 = (hotel_market)
        if s3 in best_hotels_country:
            d = best_hotels_country[s3]
            dsum = 0.01
            for k,v in d.items():
                dsum += int(v)
            for k,v in d.items():
                res[int(k)] += w4 * int(v)/dsum

        dsum = 0.01
        for i in range(len(topclasters)):
            dsum += int(topclasters[i][0])
        for i in range(len(topclasters)):
            res[i] += w5 * int(topclasters[i][0])/dsum

        topitems = np.argsort(-res)
        for i in range(5):
            if int(topitems[i]) in filled:
                continue
            if len(filled) == 5:
                break
            out.write(' {}'.format(topitems[i]))
            filled.append(int(topitems[i]))
            total2 += 1
        out.write("\n")
    out.close()
    print('Total  1: {:10d} ...'.format(total1))
    print('Total  0: {:10d} ...'.format(total0))
    print('Total 00: {:10d} ...'.format(total00))
    print('Total  2: {:10d} ...'.format(total2))


best_s00,best_s01,best_hotels_country, best_hotels_od_ulc, best_hotels_uid_miss, best_hotels_search_dest, popular_hotel_cluster = prepare_arrays_match()
gen_evaluation(best_s00, best_s01,best_hotels_country, best_hotels_search_dest, best_hotels_od_ulc, best_hotels_uid_miss, popular_hotel_cluster)
gen_submission(best_s00, best_s01,best_hotels_country, best_hotels_search_dest, best_hotels_od_ulc, best_hotels_uid_miss, popular_hotel_cluster)

