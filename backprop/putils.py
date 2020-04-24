import os
from os import listdir
from os.path import isfile, join
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def get_path_record(hs, epochs):
    path = './record/hs{}-ep{}/'.format(hs, epochs)
    record = read_files(path)
    plot_path = './plots/hs{}-ep{}/'.format(hs, epochs)
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    return record, plot_path


def read_files(path):
    record = {}
    models = [f[:-5] for f in listdir(path) if isfile(join(path, f))]
    for var in models:
        with open(path + "{}.json".format(var)) as json_file:
            record[var] = json.load(json_file)
    return record


def plot(record, metric, gap, models, opts, lrs, file_name=False):
    marker = {'sgd': 'o', 'adagrad': '+', 'adam': 'd'}
    color = {'nmodel': 'tab:red', 'wmodel': 'tab:orange',
             'bmodel': 'tab:purple'}
    ls = {'0.1': 'dotted', '0.01': 'solid', '0.001': 'dashed', '0.0001': 'dashdot'}
    for model in models:
        for opt in opts:
            for lr in lrs:
                model_name = "{}_{}_lr{}".format(model, opt, lr)
                plt.plot(record[model_name][metric][::gap], color=color[model], marker=marker[opt],
                         linestyle=ls[str(lr)], label=model_name)
    plt.legend()
    plt.xlabel("iterations/{}*100".format(gap))
    plt.ylabel(metric)
    if file_name:
        plt.savefig("{}.pdf".format(file_name))
    plt.show()
