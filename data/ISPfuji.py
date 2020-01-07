import os

from data import common
from data import ImageData

import numpy as np
import scipy.misc as misc
import glob
import torch
import torch.utils.data as data

class ISPfuji(ImageData.ImageData):
    def __init__(self, args, train=True):
        super(ISPfuji, self).__init__(args, train)
        self.repeat = args.test_every // (args.n_train // args.batch_size)


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
        return list_tar, list_input

        '''
        train_fns = glob.glob(self.dir_tar+ '/*.npy')

        train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

        for train_fn in train_fns:
            print(train_fn)
            train_id = int(os.path.basename(train_fn)[0:5])
            in_files = glob.glob(self.dir_input + '/%05d_00*a.npy' % train_id)
            gt_files = glob.glob(self.dir_tar + '/%05d_00*a.npy' % train_id)
            print(gt_files)
            for index in range(len(in_files)):
            	list_tar.append(gt_files[0])
            	list_input.append(in_files[index])
            in_files = glob.glob(self.dir_input + '/%05d_00*b.npy' % train_id)
            gt_files = glob.glob(self.dir_tar + '/%05d_00*b.npy' % train_id)
            for index in range(len(in_files)):
            	list_tar.append(gt_files[0])
            	list_input.append(in_files[index])
            in_files = glob.glob(self.dir_input + '/%05d_00*c.npy' % train_id)
            gt_files = glob.glob(self.dir_tar + '/%05d_00*c.npy' % train_id)
            for index in range(len(in_files)):
            	list_tar.append(gt_files[0])
            	list_input.append(in_files[index])
            in_files = glob.glob(self.dir_input + '/%05d_00*d.npy' % train_id)
            gt_files = glob.glob(self.dir_tar + '/%05d_00*d.npy' % train_id)
            for index in range(len(in_files)):
            	list_tar.append(gt_files[0])
            	list_input.append(in_files[index])
        '''
    def _name_tarbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_tar30.npy'.format(self.split)
        )

    def _name_inputbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_input30.npy'.format(self.split)
        )

    def _set_filesystem(self, dir_data):
        self.apath = dir_data + 'ISP/Fuji/Train'
        self.dir_tar = os.path.join(self.apath, 'longnew')
        self.dir_input = os.path.join(self.apath, 'shortnew')
        self.ext = '.npy'


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

