"""Utility functions for data exploration"""
import json

import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

sns.set(style="darkgrid")
plt.rcParams['figure.figsize'] = (6, 6)


def plot_scatter(df, vars=['width', 'height'], hue='birads',
                 logy=True, logx=True, color_palette="Set2"):
    """Plot scatter plot

    Args:
        df:
        vars: list of columns to plot, in the order of [x, y]
        hue:
        logy:
        logx:
        color_palette:

    Returns:
        None
    """
    xlabel, ylabel = vars[:2]
    hue_order = sorted(df[hue].unique())
    colors = sns.color_palette(color_palette, len(hue_order))
    # if any value is string and cannot be converted to a numerical value, plot categorically
    is_categorical = any([type(a) is str and not a.isdigit() for a in df[ylabel].unique()])
    if is_categorical:
        y_order = sorted(df[ylabel].unique())
    for idx, label in enumerate(hue_order):
        df_tmp = df[df[hue] == label]
        xs = df_tmp[xlabel]
        ys = df_tmp[ylabel]
        if is_categorical:
            ys = ys.apply(lambda x: y_order.index(x))
        plt.scatter(xs, ys, color=colors[idx], label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if logx:
        plt.xscale('log')
    if logy and not is_categorical:
        plt.yscale('log')
    if is_categorical:
        # print(list(zip(*(enumerate(hue_order)))))
        plt.yticks(*zip(*(enumerate(y_order))))


def plot_bubble(df, vars=['disease', 'birads'], color='b'):
    """Plot bubble chart for categorical data

    Args:
        df:
        vars:
        color:

    Returns:
        None
    """
    xlabel, ylabel = vars[:2]
    df_group = df.groupby([xlabel, ylabel]).count()['size']
    x_order = sorted(df[xlabel].unique())
    y_order = sorted(df[ylabel].unique())
    max_count = max(df_group)
    for idx, count in df_group.iteritems():
        count /= max_count
        x_idx, y_idx = idx
        x, y = x_order.index(x_idx), y_order.index(y_idx)
        plt.scatter(x, y, s=count * 10000, c=color, alpha=0.2, edgecolors='gray', linewidth=0)
    plt.xticks(*zip(*(enumerate(x_order))))
    plt.yticks(*zip(*(enumerate(y_order))))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Categorical Count')


def test():
    json_file = '/tmp/lesion_dict.json'
    with open(json_file, 'r') as f_in:
        lesion_dict = json.load(f_in)

    for key in list(lesion_dict.keys()):
        for idx in lesion_dict[key].keys():
            new_key = key + '-' + idx
            lesion_dict[new_key] = lesion_dict[key][idx]
        lesion_dict.pop(key)

    df = pd.DataFrame.from_dict(lesion_dict, orient='index')
    df = df.drop(['bbox'], axis=1)
    df['size'] = ((df['width'] * df['height']) ** 0.5).astype(int)

    plt.figure()
    plot_scatter(df, vars=['size', 'birads'], hue='disease')
    plt.show()

    plot_bubble(df, vars=['birads', 'disease'])
    plt.show()


if __name__ == '__main__':
    test()