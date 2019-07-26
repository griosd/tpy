import os
import dill
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import torch
from .core import numpy


class DictObj(dict):

    def __init__(self, data=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if data is not None:
            for k, v in data.items():
                self[k] = v

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def clone(self):
        return DictObj(data=self)

    def copy(self):
        return DictObj(data=self)


def DataFrame(tensor, *args, **kwargs):
    return pd.DataFrame(numpy(tensor), *args, **kwargs)


def distplot(tensor, *args, **kwargs):
    return sb.distplot(numpy(tensor), *args, **kwargs)


def plot(tensor, *args, **kwargs):
    return plt.plot(numpy(tensor), *args, **kwargs)


def plot2d(tensor1, tensor2=None, *args, **kwargs):
    if tensor2 is None:
        return plt.plot(numpy(tensor1), *args, **kwargs)
    return plt.plot(numpy(tensor1), numpy(tensor2), *args, **kwargs)


def matshow(tensor, *args, **kwargs):
    return plt.matshow(numpy(tensor), *args, **kwargs)


def pairplot(tensor, simple=False, bins='auto', labels=None, kde=True, max_samples=1000, max_dim=5, height=2.5,
             aspect=1, *args, **kwargs):
    try:
        df = DataFrame(tensor, columns=labels)
    except:
        df = tensor
    if max_dim is not None:
        df = df.T[:max_dim].T
    if max_samples is not None:
        if max_samples < len(df):
            df = df.sample(max_samples)
    if simple:
        sb.pairplot(df, *args, **kwargs)
    else:
        g = sb.PairGrid(df, height=height, aspect=aspect)
        g.map_diag(sb.distplot, kde=kde, bins=bins)
        g.map_lower(sb.kdeplot, n_levels=6)
        g.map_upper(sb.scatterplot)


show = plt.show
xlim = plt.xlim
ylim = plt.ylim


def dict_to_array(params, trace, device=None):
    if device is None:
        device = list(params.values())[0].device
    iter_order = sorted(trace.iter_stochastic_nodes(), key=lambda x: x[0])
    new_array = torch.Tensor([]).to(device)
    for y in iter_order:
        name, node = y
        new_array = torch.cat((new_array, params[name].view(-1).to(device)))
    return numpy(new_array)


def get_params_df_from_dict(params_dict, loss_dict={}, name_index='_niter'):

    first_param = list(params_dict.keys())[0]
    nparams = params_dict[first_param].shape[0]
    params_df = pd.DataFrame(index=pd.Index(range(nparams), name=name_index))
    for var, value in params_dict.items():
        for i, v in enumerate(value.transpose(0, 1).contiguous().view(-1, nparams)):
            params_df[var+str(i)] = numpy(v)

    for name, loss in loss_dict.items():
        params_df[name] = loss

    return params_df.reset_index().sort_index(axis=1)


def get_params_df_from_list_dict(params_list, loss={}):
    params_df_list = []
    for i, p in enumerate(params_list):
        df = get_params_df_from_dict(p, {k: v[:, i] for k, v in loss.items()}, name_index='_nchain')
        df['_niter'] = i
        params_df_list += [df]
    params_df = pd.concat(params_df_list).sort_values(by=['_nchain', '_niter']).reset_index().drop(['index'], axis=1).sort_index(axis=1)
    return params_df


def get_params_dict_from_df(params_df, trace, device):
    params_protoype = {name: node['value'] for name, node in trace.iter_stochastic_nodes()}
    params_dict = {}
    nparams = len(params_df)
    for var in params_protoype.keys():
        shape = list(params_protoype[var].shape)
        shape[0] = nparams
        params_dict[var] = torch.empty(shape, device=device)

    for var, value in params_dict.items():
        shape = params_dict[var].shape
        columns = [var + str(i) for i in range(shape[1])]
        values = torch.tensor(params_df[columns].values, device=device)
        params_dict[var] = values.reshape(shape)
    return params_dict


def plot_training(params_df, burnin = False, outlier = False, varnames=None, transform=lambda x: x, figsize=None,
                  lines=None, combined=False, grid=True,
                  alpha=0.35, priors=None,
                  ax=None, traces=True, quantiles=[0.1, 0.9], confidence=True):

    nburnin = params_df.loc[(~params_df['_burnin']).idxmin()]['_niter']
    times_burnin = np.arange(nburnin, params_df['_niter'].max()+1)
    # np.nonzero(params_df.groupby('_niter').mean()['_burnin'].values)[0]
    burnin_index = params_df['_burnin'].values
    if burnin and hasattr(params_df, '_burnin'):
        params_df = params_df[params_df['_burnin']]
    # if outlier and hasattr(params_df, '_outlayer'):
    #     params_df = params_df[params_df._outlayer]
    if outlier and hasattr(params_df, '_outlier'):
        params_df = params_df[params_df['_outlier']]
    params_df = params_df.set_index(['_nchain']).drop(['_burnin', '_outlier'], axis=1)
    if combined:
        params_df.index = params_df.index * 0

    if varnames is None:
        varnames = [k for k in params_df.columns if k not in ['_niter', '_nchain']]

    n = len(varnames)

    if figsize is None:
        figsize = (12, n * 2)

    if ax is None:
        fig, ax = plt.subplots(n, 2, squeeze=False, figsize=figsize)
    elif ax.shape != (n, 2):
        print('traceplot requires n*2 subplots')
        return None
    describe = params_df.describe(percentiles=[0.01, 0.99]).T
    mean_chain = params_df.groupby('_niter').mean()
    lower_q = params_df.groupby('_niter').quantile(q=quantiles[0])
    upper_q = params_df.groupby('_niter').quantile(q=quantiles[1])

    d_min = describe['1%']
    d_max = describe['99%']
    for i, v in enumerate(varnames):
        d_total = np.reshape(params_df[burnin_index][v], -1)
        try:
            sb.distplot(d_total, ax=ax[i, 0], norm_hist=True, hist=True, color='k')
        except Exception as e:
            print(v, e)

        if traces:
            for key in params_df.index.unique():
                dk = params_df[burnin_index].loc[key]
                dk = dk[np.isfinite(dk[v])]
                d = np.squeeze(transform(dk[v]))
                sb.distplot(d, ax=ax[i, 0], norm_hist=True, hist=False)
                try:
                    ax[i, 1].plot(times_burnin, d, alpha=alpha)
                except Exception as e:
                    print(v, e)

        ax[i, 0].set_xlim([d_min[v], d_max[v]])
        ax[i, 1].plot(mean_chain[v][nburnin:], color="k", alpha=0.6)
        if confidence:
            ax[i, 1].fill_between(times_burnin,
                                  np.squeeze(lower_q[v].values)[nburnin:],
                                  np.squeeze(upper_q[v].values)[nburnin:], facecolor='b', alpha=0.4)
        ax[i, 1].axvline(x=nburnin, color="k", lw=1.5, alpha=0.3)
        ax[i, 1].set_ylim([d_min[v], d_max[v]])
        if lines:
            try:
                ax[i, 0].axvline(x=lines[v], color="r", lw=1.5)
                ax[i, 1].axhline(y=lines[v][nburnin:], color="r", lw=1.5, alpha=alpha)
            except KeyError:
                pass

        # name_var = str(v)
        # ax[i, 0].set_ylabel(name_var[name_var.find('_')+1:])
        ax[i, 1].set_ylabel("Sample value")
        ax[i, 0].grid(grid)
        ax[i, 0].set_ylim(bottom=0)
    plt.tight_layout()


def set_style():
    plt.rcParams['figure.figsize'] = (20, 6)
    # output_notebook()


def plot_text(title="title", x="xlabel", y="ylabel", ncol=3, loc='best', axis=None, legend=True):
    plt.axis('tight')
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    if legend:
        plt.legend(ncol=ncol, loc=loc)
    if axis is not None:
        plt.axis(axis)
    plt.tight_layout()


def save_df(df, path='df.h5', key='df'):
    rfind = path.rfind('/')
    if rfind > 0:
        os.makedirs(path[:rfind], exist_ok=True)
    df.to_hdf(path, key=key)


def load_df(path='df.h5', key='df'):
    return pd.read_hdf(path, key)


def save_pkl(objs, path='model.pkl'):
    rfind = path.rfind('/')
    if rfind > 0:
        os.makedirs(path[:rfind], exist_ok=True)
    with open(path, 'wb') as file:
        dill.dump(objs, file)


def load_pkl(path='model.pkl'):
    with open(path, 'rb') as file:
        objs = dill.load(file)
    return objs
