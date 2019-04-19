import os
import dill
from .utils import *
from .samplers import *
from .distributions import *
from .distributions.same_copula import *
from .distributions.location_scatter import *
from .distributions.spherically_equivalent import *


def set_style():
    plt.rcParams['figure.figsize'] = (20, 6)
    #output_notebook()


def save_df(df, path='df.h5', key='df'):
    rfind = path.rfind('/')
    if rfind > 0:
        os.makedirs(path[:rfind], exist_ok=True)
    df.to_hdf(path, key=key)


def load_df(path='df.h5', key='df'):
    return pd.read_hdf(path, key)


def save_pkl(objs, path ='model.pkl'):
    rfind = path.rfind('/')
    if rfind > 0:
        os.makedirs(path[:rfind], exist_ok=True)
    with open(path, 'wb') as file:
        dill.dump(objs, file)


def load_pkl(path = 'model.pkl'):
    with open(path, 'rb') as file:
        objs = dill.load(file)
    return objs
