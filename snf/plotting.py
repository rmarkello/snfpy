# -*- coding: utf-8 -*-

import numpy as np


def mod_heatmap(arr, labels, **kwargs):
    """
    Plots ``arr`` heatmap with border around ``labels``

    Parameters
    ----------
    arr : (N x N) array_like
        Adjacency matrix to be plotted as heatmap
    labels : (N,) array_like
        Group assignments for nodes in ``arr``
    **kwargs
        Keyword arguments for `seaborn.heatmap()`

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
    np.ndarray
        Indices to sort
    """

    import seaborn as sns
    from matplotlib import patches

    arr, labels = np.array(arr), np.array(labels)
    # we can't have a label of 0 -- needs to start at 1
    if 0 in labels:
        labels += 1
    # get sorted label assignments
    inds = labels.argsort()
    # get bounds of labels for plotting outlines
    bounds = np.hstack([[0],
                        np.where(np.diff(labels[inds]))[0] + 1,
                        [len(inds)]]).astype(int)
    # sort labels by strength of node (to be pretty)
    for n, f in enumerate(bounds[:-1]):
        i = inds[f:bounds[n + 1]]
        cco = i[arr[np.ix_(i, i)].mean(axis=1).argsort()[::-1]]
        inds[f:bounds[n + 1]] = cco
    # plot heatmap with diagonal masked
    ax = sns.heatmap(arr[np.ix_(inds, inds)],
                     mask=np.eye(len(arr)), square=True, robust=True,
                     xticklabels=[], yticklabels=[], **kwargs)
    # add outlines of communities
    for n, edge in enumerate(np.diff(bounds)):
        ax.add_patch(patches.Rectangle((bounds[n], bounds[n]), edge, edge,
                                       fill=False, linewidth=2))

    return ax, inds
