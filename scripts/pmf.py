import numpy as np
import pandas 
import random
import math

from IPython.parallel import Client
from numpy.linalg import norm
from pandas import DataFrame
from numpy.random import normal

def loss(R, U, V):
    E = 0.0
    for (t, i, j, rij) in R.itertuples():
        E += (rij - np.dot(U[i], V[j]))**2  + norm(U[i]) + norm(V[j])
        
    return E 

def sgd(alpha=0.1, eta0=0.01, power_t=0.25, epochs=3, latent_dimensions=10):
    """ stochastic gradient descent """

    n_users = df['user_id'].nunique()
    n_items = df['hotel_cluster'].nunique()

    U = DataFrame(normal(size=(latent_dimensions, n_users)),
                  columns=df['user_id'].unique())

    V = DataFrame(normal(size=(latent_dimensions, n_items)),
                  columns=df['hotel_cluster'].unique())

    t = 1.0
    index = df.index.values
    random.shuffle(index)

    for epoch in xrange(epochs):

        for count, pos in enumerate(index):

            i, j, rij = df.ix[pos] 
            eta =  eta0 / (t ** power_t)

            rhat = dot(U[i], V[j])

            U[i] = U[i] - eta * ((rhat - rij) * V[j] + alpha * U[i])
            V[j] = V[j] - eta * ((rhat - rij) * U[i] + alpha * V[j])

            if isnan(U.values).any() or isnan(V.values).any():
                raise ValueError('overflow')
            
            t += 1

        return U, V

def fit(df, alpha=0.1, eta0=0.01, power_t=0.25, epochs=3,
        latent_dimensions=10):

    rc = Client()
    dview = rc[:]
    k = float(len(rc))

    with dview.sync_imports():
        import random 

        from numpy.linalg import norm
        from numpy import dot, isnan
        from numpy.random import normal
        from pandas import DataFrame

    dview.scatter('df', df)
    res = dview.apply_sync(sgd, alpha=0.1, eta0=0.01, power_t=0.25, epochs=3,
            latent_dimensions=latent_dimensions)

    add = lambda a,b: a.add(b, fill_value=0)
    U = reduce(add, (r[0] for r in res))/k
    V = reduce(add, (r[1] for r in res))/k

    return U, V

if __name__ == '__main__':

    latent_dimensions = 200
    alpha = 0.1
    eta0 = 0.01
    epochs = 5
    power_t = 0.25

    # read data and fix indexing 
    df = pandas.read_csv('../input/train_sample.csv', header=1, sep=',', 
         names=['date_time','site_name','posa_continent','user_location_country','user_location_region','user_location_city','orig_destination_distance','user_id','is_mobile','is_package','channel','srch_ci','srch_co','srch_adults_cnt','srch_children_cnt','srch_rm_cnt','srch_destination_id','srch_destination_type_id','is_booking','cnt','hotel_continent','hotel_country','hotel_market','hotel_cluster'])

    df = df[['user_id', 'hotel_cluster', 'is_booking']]

    n_users = df['user_id'].nunique()
    n_items = df['hotel_cluster'].nunique()

    U = DataFrame(normal(size=(latent_dimensions, n_users)),
                  columns=df['user_id'].unique())

    V = DataFrame(normal(size=(latent_dimensions, n_items)),
                  columns=df['hotel_cluster'].unique())
    
    print loss(df, U, V)

    U, V = fit(df, alpha=alpha, eta0=eta0, power_t=power_t, 
               epochs=epochs, latent_dimensions=latent_dimensions)

    print loss(df, U, V)
