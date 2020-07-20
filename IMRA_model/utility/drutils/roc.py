"""
This file contains common utility functions for drawing ROC curves
"""
import ast
import glob
import glob2
import json
import re
import numpy as np
import os
# matplotlib.use('Agg') # use non-interactive backend
from matplotlib import pylab as plt
from sklearn import metrics
import itertools
from projects.drutils import fileio



def get_ax(rows=1, cols=1, size=(16, 16)):
    """Quick control of fig size and layout

    Return a Matplotlib Axes array to be used in all visualizations in the notebook.
    Provide a central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size[0] * cols, size[1] * rows))
    return ax


def find_threshold(fpr, tpr, thresholds, target_fpr):
    """
    Find the threshold corresponding to a target FPR (target_fpr)
    Args:
        fpr: List of FPR
        tpr: List of TPR
        thresholds: List of thresholds
        target_fpr: Target FPR at which to operate

    Returns:
        target_thr: Threshold that produces the target FPR
        target_tpr: TPR at the target FPR
    """
    assert(len(fpr) == len(thresholds))
    fpr = np.asarray(fpr)
    thresholds = np.asarray(thresholds)

    # Find index such that fpr[idx-1] < target_fpr < fpr[idx]
    idx = fpr.searchsorted(target_fpr)
    # print('idx=', idx)
    if idx == len(fpr):
        print("Target FPR out of range. Maximum FPR={} at threshold={}".format(fpr[-1], thresholds[-1]))
        target_thr = thresholds[-1]
    elif idx == 0:
        print("Target FPR out of range. Minimum FPR={} at threshold={}".format(fpr[0], thresholds[0]))
        target_thr = thresholds[0]
    else:
        left_fpr = fpr[idx-1]
        right_fpr = fpr[idx]
        interpolation_frac = (target_fpr - left_fpr) / (right_fpr - left_fpr)

        left_tpr = tpr[idx-1]
        right_tpr = tpr[idx]
        target_tpr = left_tpr + (right_tpr - left_tpr) * interpolation_frac

        left_thr = thresholds[idx-1]
        right_thr = thresholds[idx]
        target_thr = min(1.0, max(0.0, left_thr + (right_thr - left_thr) * interpolation_frac))
    return target_thr, target_tpr


def plot_crosshair(coordinates, ax=None, **kwargs):
    """
    Plot crosshair at target cordinate
    Args:
        coordinates: the x, y coordinates of the point to be plotted
    Return:
        crosshair_handles: handles to crosshair lines
    """
    x, y = coordinates
    if ax is None:
        ax = plt.gca()
    horiz = ax.axhline(y, **kwargs)
    vert = ax.axvline(x, **kwargs)
    annotation = '({:.2f},{:.2f})'.format(x, y)
    plt.annotate(annotation, (x + 0.01, y - 0.04), color=kwargs['color'])
    crosshair_handles = horiz, vert
    return crosshair_handles


def plot_inset(fig, auc_list, title='auc', xlabel='ckpt', location=[0.55, 0.25, 0.3, 0.3]):
    left, bottom, width, height = location
    ax1 = fig.add_axes([left, bottom, width, height])
    ax1.plot(auc_list, color='k', marker='o')
    plt.xlabel(xlabel)
    plt.title(title)


def plot_roc_from_txt(fig, filename, idx=0, target_fpr=0.5, show_crosshair=True, plot_type='plot', title=''):
    """ Plot ROC curve from a text file
    Args:
        filename: Each line of the text file contains a prediction in [0, 1] and a label, separated by comma
        idx: optional, index number of the current curve
    Return:
        auc: Area under ROC curve
    """
    # get predictions and labels lists
    preds = []
    labels = []
    image_ids = []
    with open(filename, 'r') as infile:
        for line in infile:
            items = [item.strip() for item in line.split(',')]
            pred, label = items[0], items[1]
            preds.append(float(pred))
            labels.append(int(label))
    preds = np.array(preds)
    labels = np.array(labels)
    num_neg_label = (labels == 0).sum()
    # plot/add to ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, drop_intermediate=False)

    target_thr, target_tpr = find_threshold(fpr, tpr, thresholds, target_fpr)
    auc = metrics.auc(fpr, tpr)
    data_label = '{}. {}: {:.3f}'.format(idx, os.path.basename(filename), auc)
    plt.figure(fig.number)
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    xs = fpr
    if plot_type == 'plot':
        plt.plot(xs, tpr, label=data_label)
    elif plot_type == 'scatter':
        plt.scatter(xs, tpr, s=80, facecolors='none', edgecolors='b', marker='o', label=data_label)
    # plt.tight_layout()
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.grid(linestyle=':')
    
    lgd = plt.legend(loc='center left', fontsize=12, bbox_to_anchor=(1, 0.5))
    plt.xlabel('FPR', fontsize=12)
    plt.ylabel('TPR (Recall)', fontsize=12)
    plt.title(title, fontsize=18)

    if show_crosshair:
        coordinates = (target_fpr, target_tpr)
        plot_crosshair(coordinates, color='red', lw=1, linestyle='--')
        disp_fpr = [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5]
        disp_thr = [0.0] * len(disp_fpr)
        disp_tpr = [0.0] * len(disp_fpr)

        annotation = '({:4},{:4},{:4})'.format('FPR', 'TPR', 'Thr')
        x, y = coordinates
        # Move annotation to top if curve is too low
        if y < 0.5:
            y = 1.0
        plt.annotate(annotation, (x + 0.01, y - 0.12))
        for i in range(len(disp_fpr)):
            disp_thr[i], disp_tpr[i] = find_threshold(fpr, tpr, thresholds, disp_fpr[i])
            print("FPR={}, TPR={:.4f} at threshold={:.4f}".format(disp_fpr[i], disp_tpr[i], disp_thr[i]))
            annotation = '({:.2f},{:.2f},{:.3f})'.format(disp_fpr[i], disp_tpr[i], disp_thr[i])
            plt.annotate(annotation, (x + 0.01, y - 0.12 - 0.04*(1+i)))

    return auc, lgd


def split_with_square_brackets(input_str):
    """
    Split a string using "," as delimiter while maintaining continuity within "[" and "]"
    Args:
        input_str: Input string

    Returns:
        substrings: List of substrings
    """
    substrings = []
    bracket_level = 0
    current_substr = []
    for next_char in (input_str + ","):
        if next_char == "," and bracket_level == 0:
            substrings.append("".join(current_substr))
            current_substr = []
        else:
            if next_char == "[":
                bracket_level += 1
            elif next_char == "]":
                bracket_level -= 1
            current_substr.append(next_char)
    return substrings


def update_preds_and_labels(pred, labels_matched, image_id, detected_labels, preds, labels, image_ids):
    """
    Convert the matched labels of the prediction bbox to binary label and update all predictions and labels
    Args:
        pred: Current prediction score
        labels_matched: GT labels for which the current prediction is a match
        image_id: Current image ID
        detected_labels: GT labels that have been detected so far for this image_id
        preds: List of all predictions, passed by reference
        labels: List of all binary labels, passed by reference
        image_ids: List of all image IDs, passed by reference

    Returns:
        detected_labels: GT labels that have been detected so far for this image_id
    """
    num_detected_so_far = len(detected_labels)
    detected_labels = detected_labels.union(set(labels_matched))

    # If the current prediction contribute to a new match, set label to 1
    if len(detected_labels) > num_detected_so_far:
        label = 1
        for _ in range(len(detected_labels) - num_detected_so_far):
            preds.append(float(pred))
            labels.append(float(label))
            image_ids.append(image_id)
    else:
        label = 0
        preds.append(float(pred))
        labels.append(float(label))
        image_ids.append(image_id)

    return detected_labels


def plot_froc_from_txt(fig, filename, idx=0, target_fpr=0.5, show_crosshair=True, plot_type='plot', title=''):
    """ Plot FROC curve from a text file
    Args:
        filename: Each line of the text file contains a prediction in [0, 1] and a label, separated by comma
        idx: optional, index number of the current curve
    Return:
        auc: Area under ROC curve
    """
    # get predictions and labels lists
    preds = []
    labels = []
    image_ids = []
    with open(filename, 'r') as infile:
        for line in infile:
            # items = [item.strip() for item in line.split(',')]
            items = [item.strip() for item in split_with_square_brackets(line)]
            pred, labels_matched = items[0], items[1]
            labels_matched = ast.literal_eval(labels_matched)
            try:
                image_id = items[2]
            except:
                raise ValueError('Every line must have image_id for FROC curve generation!')
            if float(pred) == 0 and len(labels_matched) == 0:
                continue
            if image_id not in image_ids:
                detected_labels = set()
            detected_labels = update_preds_and_labels(pred, labels_matched, image_id, detected_labels, \
                                                      preds, labels, image_ids)
    preds = np.array(preds)
    labels = np.array(labels)
    num_unique_image_ids = len(set(image_ids))
    num_neg_label = (labels == 0).sum()
    # plot/add to ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
    fpr, tpr, thresholds = fpr[:-1], tpr[:-1], thresholds[:-1]
    # this is equivalent to [(labels[preds > threshold] == 0).sum() / num_image_ids for threshold in thresholds]
    neg_per_image = num_neg_label / num_unique_image_ids
    fpc = fpr * neg_per_image
    
    # print 'fpr, tpr, thresholds:
    target_thr, target_tpr = find_threshold(fpr, tpr, thresholds, target_fpr)
    auc = metrics.auc(fpr, tpr)
    data_label = '{}. {}: {:.3f}'.format(idx, os.path.basename(filename), auc)
    plt.figure(fig.number)
    xs = fpc
    if plot_type == 'plot':
        plt.plot(xs, tpr, label=data_label)
    elif plot_type == 'scatter':
        plt.scatter(xs, tpr, s=80, facecolors='none', edgecolors='b', marker='o', label=data_label)
    plt.tight_layout()
    plt.xlim([0, 10.0])
    plt.ylim([0, 1.0])
    plt.grid(linestyle=':')

    lgd = plt.legend(loc='center left', fontsize=12, bbox_to_anchor=(1, 0.5))
    plt.xlabel('FP per Image', fontsize=12)
    plt.ylabel('TPR (Recall)', fontsize=12)
    plt.title(title, fontsize=18)

    if show_crosshair:
        coordinates = (target_fpr * neg_per_image, target_tpr)
        plot_crosshair(coordinates, color='red', lw=1, linestyle='--')

        disp_fpr = [0.001, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.04]
        disp_thr = [0.0] * len(disp_fpr)
        disp_tpr = [0.0] * len(disp_fpr)

        annotation = '({:4},{:4},{:4})'.format('FPPI', 'TPR', 'Thr')
        x, y = coordinates
        plt.annotate(annotation, (x + 0.01, y - 0.12))
        for i in range(len(disp_fpr)):
            disp_thr[i], disp_tpr[i] = find_threshold(fpr, tpr, thresholds, disp_fpr[i])
            annotation = '({:.2f},{:.2f},{:.2f})'.format(disp_fpr[i] * neg_per_image, disp_tpr[i], disp_thr[i])
            plt.annotate(annotation, (x + 0.01, y - 0.12 - 0.04*(1+i)))

    return auc, lgd


def plot_froc_from_data_dict(data_dict, output_fig_path=None, fig_title=None,
                             label_filter='', xlim=(0.1, 50), key_sorter=None, plot_recall=False,
                             highlight_idx=None, **kwargs):
    """Plot froc curve from a data dict

    Args:
        data_dict: a dict of dict. Each sub-dict has keys
            label: used as legend
            data: list of list in the format [[recall, fpc, threshold], ...]
        output_fig_path:
        fig_title:
        label_filter:
        xlim:
        key_sorter: a function to sort the keys. Default to sorting by last mod time
        plot_recall: defaults to False, where a semilogx and a linear plot are plotted side by side.
            When plot_recall is True, replcae the linear plot with plot of recall in chronological order
        highlight_idx: the idx (counting from 0) of the threshold list to plot trend over time

    Returns:
        None
    """
    fig = plt.figure(figsize=(12, 6))
    labels = sorted(set(val['label'] for val in data_dict.values()))
    line_styles = ['-', ':', '-.', '--']

    if len(labels) > len(line_styles) or len(data_dict) == len(labels):
        ls_dict = {label:line_styles[0] for idx, label in enumerate(labels)}
    else:
        ls_dict = {label:line_styles[idx] for idx, label in enumerate(labels)}
    if plot_recall:
        plot_fns = [plt.semilogx]
    else:
        plot_fns = [plt.semilogx, plt.plot]
    for idx, plot_func in enumerate(plot_fns):
        plt.subplot(1, 2, idx+1)
        keys = sorted(data_dict.keys())
        # sort by last mod time
        key_sorter = key_sorter or (lambda x: os.path.getmtime(x))
        try:
            keys.sort(key=key_sorter)
        except:
            # if cannot sort with key_sorter, sort alphabetically by key string
            keys.sort(key=str)
        mid_recall_list = []
        mid_fp_list = []
        for key in keys:
            label = data_dict[key]['label']
            line_style = ls_dict[label]
            if 'num_images' in data_dict[key]:
                label = '{} (count:{})'.format(label, data_dict[key]['num_images'])
            if label_filter in label:
                data = data_dict[key]['data']
                fpc = [item[1] for item in data]
                recall = [item[0] for item in data]
                p = plot_func(fpc, recall, marker='.', ls=line_style, label=label)
                color = p[0].get_color()
                if highlight_idx is not None:
                    if highlight_idx == 'mid':
                        highlight_idx = (len(fpc) - 1) // 2
                    mid_recall_list.append(recall[highlight_idx])
                    mid_fp_list.append(fpc[highlight_idx])
                    plt.scatter(fpc[highlight_idx], recall[highlight_idx],
                                marker='o', s=100, facecolors='none', edgecolors=color)
        plt.xlabel('FP per Image')
        plt.ylabel('Recall')
        plt.title(fig_title)
        plt.xlim(xlim)
        plt.ylim([0, 1])
        plt.grid()
        plt.yticks([i / 10.0 for i in range(0, 10)])
        plt.grid(b=True, which='major', color='gray', linestyle='-')
        plt.grid(b=True, which='minor', color='gray', linestyle='--')
    # only plot the legend of the last subplot
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    if plot_recall:
        # plot on the RHS
        ax1 = plt.subplot(122)
        ax2 = ax1.twinx()
        plot_sharex_series(ax1, ax2, mid_recall_list, mid_fp_list, ylim1=(0, 1), ylim2=(0.1, 10),
                           xlabel='ckpt', ylabels=('Recall', 'FPC'))
        plt.grid()
        plt.title('Recall and FPC')
    if output_fig_path is not None:
        fileio.maybe_make_new_dir(os.path.dirname(output_fig_path))
        plt.savefig(output_fig_path, dpi=300) # default dpi is usually 100
        plt.close('all')
    else:
        plt.show()
    return fig


def plot_sharex_series(ax1, ax2, data1, data2, t=None, ylim1=None, ylim2=None,
                       xlabel='', ylabels=('', ''), colors=('tab:red', 'tab:blue')):
    """"Plot two data series of different scales on the same graph

    Adapted from https://matplotlib.org/gallery/api/two_scales.html
    """
    color1, color2 = colors
    ylabel1, ylabel2 = ylabels
    if not t:
        t = range(1, len(data1) + 1)
    # plot first series
    color = color1
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color=color)
    ax1.plot(t, data1, color=color, ls='--', marker='o', markersize=10, markerfacecolor='none')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yticks([i / 10.0 for i in range(0, 10)])
    ax1.yaxis.grid(color=color, linestyle='--')
    ax1.xaxis.grid(color='gray', linestyle='--')
    if ylim1:
        ax1.set_ylim(ylim1)
    # plot second series
    color = color2
    ax2.set_ylabel(ylabel2, color=color)  # we already handled the x-label with ax1
    ax2.semilogy(t, data2, color=color, linestyle='--', marker='o', markersize=10, markerfacecolor='none')
    ax2.tick_params(axis='y', labelcolor=color)
    if ylim2:
        ax2.set_ylim(ylim2)
    ax2.yaxis.grid(color='black', linestyle='--') # <FIXME> this does not show up. Why?
    plt.tight_layout()  # otherwise the right y-label is slightly clipped


def batch_plot_froc_json(input_search_path, output_fig_path=None, name='', legend_regex_sub='', **kwargs):
    """Plot json in a directory onto one froc

    Args:
        input_search_path: glob pattern, such as '/data/log/mammo/calc_train/Mammo_20180318-22h44PM39/froc*json', so
            it could be a path to a specific file. It could also be a list of glob patterns, but they should have
            the same parent (FROC title uses the parent folder of the first pattern).
        output_fig_path:
        name: FROC dataset patterns in title
        legend_regex_sub: regex pattern to delete from legend labels

    Returns:
        None
    """
    if not isinstance(input_search_path, (list, tuple)):
        input_search_path = [input_search_path]
    froc_json_path_list = []
    for single_search_path in input_search_path:
        froc_json_path_list.extend(glob2.glob(single_search_path))
    froc_json_path_list = sorted(set(froc_json_path_list))
    # generate fig title
    input_dir = os.path.dirname(input_search_path[0])
    json_dirname = os.path.basename(input_dir.strip(os.sep))  # the last level of folder name
    fig_title = '{} FROC {}'.format(name, json_dirname)
    data_dict = {}
    for froc_json_path in froc_json_path_list:
        with open(froc_json_path, 'r') as f_in:
            data_dict[froc_json_path] = {}
            label = os.path.basename(froc_json_path).replace('.json', '')#.replace(legend_str_omit, '')
            label = re.sub(legend_regex_sub, '', label)
            data_dict[froc_json_path]['label'] = label
            data_dict[froc_json_path]['data'] = json.load(f_in)
    plot_froc_from_data_dict(data_dict, output_fig_path, fig_title, **kwargs)

