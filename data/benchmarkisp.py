import os

from data import common
from data import ImageData

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class BenchmarkISP(ImageData.ImageData):
    def __init__(self, args, name = '', train=True):
        super(BenchmarkISP, self).__init__(args, name=name , train = train, benchmark=True)

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'BenchmarkISP', self.args.data_test,'Test')

        self.dir_tar = os.path.join(self.apath, 'longnew')
        self.dir_input = os.path.join(self.apath, 'shortnew')
        self.ext = '.npy'
        print(self.dir_tar)

    def _name_tarbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_tar.npy'.format(self.split)
        )

    def _name_inputbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_input.npy'.format(self.split)
        )




    def _scan(self):
        list_tar = []
        list_input = []
        inputfiles = self.dir_input + '/inputfiles.txt'
        tarfiles   = self.dir_tar + '/targetfiles.txt'
        with open(inputfiles, 'r') as filehandle:  
            filecontents = filehandle.readlines()

            for line in filecontents:
            	current_place = line[:-1]
            	list_input.append(current_place)

        with open(tarfiles, 'r') as filehandle:  
            filecontents = filehandle.readlines()

            for line in filecontents:
            	current_place = line[:-1]
            	list_tar.append(current_place)
        #print(list_tar)
        return list_tar, list_input






    def __len__(self):
        if self.train:
            return len(self.images_tar) * self.repeat
        else:
            return len(self.images_tar)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_tar)
        else:
            return idx

