import numpy as np
import os
import json
import time
from tqdm import tqdm
import torch

from utils.ts2vec import TS2Vec
from utils.conf import Configuration
import tasks

# MODEL_NAME = ['seanet', 'tsrnet']
MODEL_NAME = ['seanet']
# MODEL_NAME = ['tsrnet']


def train():
    currtime = time.strftime('%Y%m%d%H%M%S', time.localtime())

    # confs_path = 'ts2vec/datasets/test_confs'
    confs_path = 'ts2vec/datasets/all_confs'
    # confs_path = 'ts2vec/datasets/confs_air_sensor'
    # confs_path = 'ts2vec/datasets/confs_air_pollutant_50'

    confs = os.listdir(confs_path)

    for conf in tqdm(confs):
        conf_path = os.path.join(confs_path, conf)

        for model_name in MODEL_NAME:
            conf = Configuration(
                "", model_name, conf_path, currtime, 'seanet_ucr')
            experiment = TS2Vec(conf)
            experiment.run()


def eval(protocol="linear"):
    res_path = 'ts2vec/seanet_ucr/acc' + protocol + '.json'

    confs_path = 'ts2vec/datasets/all_confs'
    confs = os.listdir(confs_path)

    for conf in tqdm(confs):
        dataset_name = conf.split('.')[0]
        conf_path = os.path.join(confs_path, conf)
        with open(conf_path, 'r') as f:
            conf = json.load(f)
        dim_series = conf['dim_series']
        eval_res_one_dataset = []

        for model_name in MODEL_NAME:
            model_path = 'ts2vec/seanet_ucr/' + \
                model_name + '/model/' + dataset_name + '-model.pkl'
            if os.path.exists(model_path):
                model = torch.load(model_path)
                # print('parameters: ',  sum(x.numel() for x in model.parameters()))

                # 'classification':
                out, acc, auprc = tasks.eval_classification(
                    model, model_name, dataset_name, dim_series, eval_protocol="linear")
                print('Evaluation result of {} on {}: acc: {}, auprc: {}'.format(
                    dataset_name, model_name, acc, auprc))

                eval_res_one_dataset.append(acc)
                eval_res_one_dataset.append(auprc)
            else:
                eval_res_one_dataset.append(0)
                eval_res_one_dataset.append(0)

        # append the results to file
        with open(res_path, 'r') as f:
            load_res = json.load(f)

        load_res.update({dataset_name: eval_res_one_dataset})

        with open(res_path, 'w') as f:
            json.dump(load_res, f)


if __name__ == '__main__':
    # train()
    eval(protocol="linear")
