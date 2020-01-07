import os

from data import common

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class ImageData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.sigma = args.noise_sigma
        self.testbin = args.testbin
        self._set_filesystem(args.dir_data)

        def _load_benchmark_bin():
            self.images_tar = np.load(self._name_tarbin())
            self.images_input = np.load(self._name_inputbin())

        def _load_bin():
            self.images_tar = np.load(self._name_tarbin())
            self.images_input = np.load(self._name_inputbin())

        print('initial image data now!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('benchmark',self.benchmark)
        if self.benchmark and self.testbin:
            print('LOAD BENCHMARK BIN')
            _load_benchmark_bin()
            print(self.images_input.shape)
            print('BIN LOAD SUCCESSED!')
        elif self.benchmark:
            print('BenchmarkScaning')
            self.images_tar, self.images_input = self._scan()
            #print(self.images_tar)
            print('Scan finished!')
        elif args.ext == 'img':
            self.images_tar, self.images_input = self._scan()
        elif args.ext.find('sep') >= 0:

            print('TrainingDataScaning')
            self.images_tar, self.images_input = self._scan()
            print('Scan finished!')
            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                for v in self.images_tar:
                    img_tar = misc.imread(v)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, img_tar)
                for v in self.images_input:
                    img_input = misc.imread(v)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, img_input)
            if(self.ext=='.png'):
                self.images_tar = [v.replace(self.ext, '.npy') for v in self.images_tar            ]
                self.images_input = [v.replace(self.ext, '.npy') for v in self.images_input]

        elif args.ext.find('bin') >= 0:
            print('bibbiibibibibibibibibibbibibibibibibibibi')
            try:
                if args.ext.find('reset') >= 0:
                    raise IOError
                print('Loading a binary file')
                _load_bin()
            except:
                print('Preparing a binary file')
                bin_path = os.path.join(self.apath, 'bin')
                if not os.path.isdir(bin_path):
                    os.mkdir(bin_path)

                list_tar, list_input = self._scan()
                img_tar = [misc.imread(f) for f in list_tar]
                np.save(self._name_tarbin(), img_tar)
                del img_tar

                img_input = [misc.imread(f) for f in list_input]
                np.save(self._name_inputbin(), img_input)
                del img_input

                _load_bin()
        elif args.ext.find('memmap') >= 0:
            print(self._sample_number(),self._height_input(),self._width_input(),self._c_in())
            a = np.memmap(self._name_inputmap(),dtype='float32',mode='r',shape=(self._sample_number(),self._height_input(),self._width_input(),self._c_in()))
            self.images_tar = np.memmap(self._name_targetmap(),dtype='float32',mode='r',shape=(self_sample_.number(),self._height_target(),self._width_target(),self._c_out()))

        else:
            print('Please define data type')

    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def _name_tarbin(self):
        raise NotImplementedError

    def _name_inputbin(self, scale):
        raise NotImplementedError




    def __getitem__(self, idx):

        img_input, img_tar = self._load_file(idx)
        img_input, img_tar = common.set_channel([img_input, img_tar], self.args.n_colors)
        img_input, img_tar = self._get_patch(img_input, img_tar)
        input_tensor, tar_tensor = common.np2Tensor([img_input, img_tar], self.args.rgb_range)
        return input_tensor, tar_tensor

    def __len__(self):
        return len(self.images_tar)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):

        idx = self._get_index(idx)
        if self.benchmark and self.testbin:
            img_input = self.images_input[idx]
            img_tar = self.images_tar[idx]
        elif self.benchmark:
            img_input = np.load(self.images_input[idx])
            img_tar = np.load(self.images_tar[idx])
        elif self.args.ext == 'img':
            img_input = misc.imread(self.images_input[idx])
            img_tar = misc.imread(self.images_tar[idx])
        elif self.args.ext.find('sep') >= 0:
            img_input = np.load(self.images_input[idx])
            img_tar = np.load(self.images_tar[idx])
        elif self.args.ext.find('bin') >= 0:
            img_input = self.images_input[idx]
            img_tar = self.images_tar[idx]
        elif self.args.ext.find('memmap') >= 0:
            img_input = self.images_input[idx,:,:,:]
            img_tar = self.images_tar[idx,:,:,:]
        return img_input, img_tar

    def _get_patch(self, img_input, img_tar):

        patch_size = self.args.patch_size
        scale = self.scale
        if self.train:
            img_input, img_tar = common.get_patch(img_input, img_tar, patch_size, scale)
            img_input, img_tar = common.augment([img_input, img_tar])
            img_input = common.add_noise(img_input, self.sigma)
        else:
            ih, iw = img_input.shape[0:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale]

        return img_input, img_tar



