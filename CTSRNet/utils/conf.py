import os
import time
import json


class Configuration:
    def __init__(self, aim: str, model_name: str = 'seanet', path: str = '', exp_time: str = '', name: str = 'default') -> None:
        self.defaults = {
            'dim_series': 60,
            'kernel_size': 3,
            'num_en_resblock': 3,
            'num_de_resblock': 2,
            'num_en_channels': 256,
            'num_de_channels': 256,
            'dim_embedding': 16,
            'dim_en_latent': 256,
            'dim_de_latent': 256,

            'device': 'cuda',
            'num_epoch': 100,

            'batch_size': 2,
            'data_size': 300,
            'dataset_name': 'SyntheticControl',

            'lsuv_size': 2000,
            'lsuv_mean': 0,
            'lsuv_std': 1.,
            'lsuv_std_tol': 0.1,
            'lsuv_maxiter': 10,
            'lsuv_ortho': True,

            'momentum': 0.9,
            'lr_mode': 'linear',
            'lr_cons': 1e-3,
            'lr_max': 1e-3,
            'lr_min': 1e-5,
            'wd_mode': 'fix',
            'wd_cons': 1e-4,
            'wd_max': 1e-4,
            'wd_min': 1e-8,
            'recons_weight': 1/4,

            'name': 'seanet_round',
            'model_name': 'seanet',

            'is_embed': True,
            'task_type': 'classification'
        }
        self.settings = {}

        if path != '':
            self.loadConf(path)

        self.__setup(aim, model_name, exp_time, name)

    def __setup(self, aim, model_name, exp_time, name) -> None:
        print('conf set up.')

        self.setHP('aim', aim)
        self.setHP('model_name', model_name)
        self.setHP('currtime', exp_time)
        self.setHP('name', name)

        currtime = self.getHP('currtime')
        dataset_name = self.getHP('dataset_name')
        dim_series = self.getHP('dim_series')
        dim_embedding = self.getHP('dim_embedding')

        # result_root = os.path.join(
        #     os.getcwd(), 'ts2vec', self.getHP('name'), str(currtime), self.getHP('model_name'))
        result_root = os.path.join(
            os.getcwd(), 'ts2vec', self.getHP('name'), self.getHP('model_name'))
        os.makedirs(result_root, exist_ok=True)
        embedding_root = os.path.join(result_root, 'embedding')
        model_root = os.path.join(result_root, 'model')
        log_root = os.path.join(result_root, 'log')
        os.makedirs(embedding_root, exist_ok=True)
        os.makedirs(model_root, exist_ok=True)
        os.makedirs(log_root, exist_ok=True)

        # sample_root = os.path.join(result_root, 'samples')

        # train_filename = '-'.join([dataset_name, str(dim_series)]) + '.bin'
        # self.setHP('train_path', os.path.join(sample_root, train_filename))

        # self.setHP('train_path', os.path.join(os.getcwd(), 'ts2vec/datasets/UCR', dataset_name, dataset_name + 'TRAIN.tsv'))
        # self.setHP('val_path', os.path.join(os.getcwd(), 'ts2vec/datasets/UCR', dataset_name, dataset_name + 'TEST.tsv'))

        embedding_prefix = '-'.join([dataset_name,
                                    str(dim_series), str(dim_embedding), str(currtime)])
        self.setHP('db_embedding_path', os.path.join(
            embedding_root, dataset_name + '.npy'))
        self.setHP('model_path', os.path.join(
            model_root, dataset_name + '-model.pkl'))
        self.setHP('log_path', os.path.join(
            log_root, dataset_name + '-fit.log'))

    def getHP(self, name: str):
        if name in self.settings:
            return self.settings[name]
        if name in self.defaults:
            return self.defaults[name]
        raise ValueError('hyperparameter {} doesn\'t exist.'.format(name))

    def setHP(self, key: str, value):
        self.settings[key] = value

    def getDilation(self, depth: int) -> int:
        return int(2 ** (depth - 1))

    def loadConf(self, path: str) -> None:
        with open(path, 'r') as f:
            loaded = json.load(f)
            self.settings = loaded
