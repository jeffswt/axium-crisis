
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import sys
import time

import numpy
import torch
import torch.nn
import torch.optim
import torch.nn.functional
import torch.utils.data
import torch.utils.tensorboard
import torchvision.transforms
import PIL.Image


__all__ = [
    'get_device',  # get pytorch device
    'flush_cache',  # flush device cache
    'InoRI',  # random number generator manager
    'HikarI',  # history writer
    'TairitsU',  # history reader
    'ShirabE',  # dataset
    'KuroyuKI',  # trainer
    'train',  # trainer invoker
]


def get_device(use_cpu=False):
    """ get_device(use_cpu=) -- get pytorch device
    @param use_cpu: True if not using cuda
    @return device: pytorch device """
    try:
        assert not use_cpu
        device = torch.device('cuda')
        x = torch.zeros((1, 1), device=device)
    except Exception:
        device = torch.device('cpu')
    return device


def flush_cache(device):
    """ clean device memory cache """
    if device == torch.device('cuda'):
        torch.cuda.empty_cache()
    return


def regularize_path(*args):
    """ regularize_path(...) -- concat path and eliminate artifacts """
    path = list(str(i) for i in args)
    if len(path) == 0:
        raise AttributeError('must have path objects')
    is_root = path[0].startswith('/')
    path = ''.join('/' + str(i) + '/' for i in args)
    path = re.sub(r'/+', r'/', re.sub(r'\\', r'/', path))[1:-1]
    return ('/' if is_root else '') + path


def debug_log(*args):
    data = ' '.join(str(i) for i in args)
    print(data, end='\n')
    sys.stdout.flush()
    return


class InoRI:
    """ InoRI: global seed manager
    Same master seeds yield the same sequence of secondary seeds, thus provides
    similar or same results when using the same secondary seed, or the same
    master seed and the same yielding sequence. """

    def __init__(self, seed=None, history=None, history_id=''):
        """ InoRI(seed=, history=, history_id=)
        @param seed: master seed used to initialize all RNGs
        @param history: if recording history is necessary, pass HikarI in
        @param history_id: string id used when recording history """
        self._module_version = 'InoRI/20191116.a'
        self._master_seed = ''
        self._seed_step = ''
        self._history = history
        self._history_id = history_id
        self._memorize('init', self._module_version)
        self.master_seed(seed=seed)
        return

    def _memorize(self, *args):
        """ _memorize(...) -- write history to HikarI """
        if self._history:
            args = [self._history_id] + list(args)
            self._history.memorize(*args)
        return

    def recall_filter(self, history):
        """ recall_filter(history) -- filter recall history """
        return

    def recall(self, cnt, timestamp, *args):
        """ recall(cnt, timestamp, ...) -- execute history traceback """
        if cnt == -1:
            return
        if args[0] == 'init':
            if args[1] != self._module_version:
                raise RuntimeError('module version mismatch: %s and %s (cur)' %
                                   (args[1], self._module_version))
        elif args[0] == 'master-seed':
            self.master_seed(args[1])
        return

    def hook_recall(self, tairitsu, history_id):
        """ hook_recall(tairitsu, history_id) -- attach self to recaller """
        tairitsu.register_handler(history_id, self)
        return

    def system_seed(self, seed):
        """ system_seed(seed) -- set global seed
        @param seed: to apply this seed to random, numpy and torch
        @note this function might be device-dependent """
        if not isinstance(seed, int):
            seed = hashlib.sha256(str(seed).encode('utf-8')).hexdigest()
            seed = abs(int(seed, base=16)) % (2 ** 32)
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return

    def master_seed(self, seed=None):
        """ master_seed(seed=) -- set InoRI seeder's master seed
        @param seed: use this to generate global seeds, random if not given """
        if not seed:
            seed = hex(int(hashlib.sha256(os.urandom(256)).hexdigest(),
                       base=16))[-16:]
        seed = str(seed)
        self._memorize('master-seed', seed)
        seed = hashlib.sha256(seed.encode('utf-8', 'ignore')).hexdigest()
        seed = seed[-16:]
        self._master_seed = seed
        self._seed_step = seed[-8:]
        self.seed()
        return

    def seed(self, seed=None):
        """ seed(seed=) -- flush global seed using InoRI's master seed
        @param seed: use this to force set the secondary seed, elsewise would
                     continue global seed generation
        @return seed: current secondary seed generated """
        if seed is None:
            tmp = self._master_seed + self._seed_step
            tmp = hashlib.sha256(tmp.encode('utf-8')).hexdigest()[-8:]
            self._seed_step = tmp
        else:
            self._seed_step = seed
        self.system_seed(self._seed_step)
        return self._seed_step

    def get_state(self):
        """ get_state() -- retrieve all rng states
        @return states: rng states, {'random': ..., 'numpy': ..., 'torch': ...}
        @note this might be uninterchangable between instances """
        states = {
            'random': random.getstate(),
            'numpy': numpy.random.get_state(),
            'torch': torch.get_rng_state(),
        }
        return states

    def set_state(self, states):
        """ set_state(states) -- reset rng states
        @param states: result returned by get_state() function
        @note this might be uninterchangable between instances """
        random.setstate(states['random'])
        numpy.random.set_state(states['numpy'])
        torch.set_rng_state(states['torch'])
        return
    pass


class HikarI:
    """ HikarI -- Experiment history record manager.
    EASY: Recollection Rate loss reduced for 'LOST' notes. """

    def __init__(self, target_path, log_path=None):
        """ HikarI(target_path, log_path=None):
        @param target_path: memory output path
        @param log_path: tensorboard log path, auxillary setting """
        self._module_version = 'HikarI/20191116.a'
        self._output_path = regularize_path(target_path)
        shutil.rmtree(self._output_path, ignore_errors=True)
        os.mkdir(self._output_path)
        # Logger
        self._writer_path = regularize_path(self._output_path, 'history.db')
        if os.path.exists(self._writer_path):
            os.remove(self._writer_path)
        self._writer_base = sqlite3.connect(self._writer_path)
        self._writer = self._writer_base.cursor()
        self._writer.execute("""
            CREATE TABLE logs (
                checkpoint  INT,
                entry       TEXT
            );""")
        self._writer_checkpoint = -1
        self._writer_flush_time = time.time()
        self.memorize('__main__', 'init', self._module_version)
        # Tensorboard writer
        self._tb_writer_path = log_path
        self._tb_writer = None
        if self._tb_writer_path:
            os.mkdir(self._tb_writer_path)
            self._tb_writer = torch.utils.tensorboard.SummaryWriter(
                self._tb_writer_path)
        self._tb_writer_co_path = regularize_path(
            self._output_path, 'tensorboard/')
        os.mkdir(self._tb_writer_co_path)
        self._tb_writer_co = torch.utils.tensorboard.SummaryWriter(
            self._tb_writer_co_path)
        # Blobs
        self._blob_path = regularize_path(self._output_path, 'blobs/')
        os.mkdir(self._blob_path)
        return

    def _is_valid_epoch_to_save(self, epoch):
        """ _is_valid_epoch_to_save(epoch) -- detect whether epoch worth saving
        @param epoch: epoch number
        @return is_valid: if this epoch's model is worth saving """
        if epoch < 0:
            return False
        if epoch <= 10:
            return True
        elif epoch <= 50:
            return epoch % 5 == 0
        elif epoch <= 100:
            return epoch % 10 == 0
        return epoch % 50 == 0

    def memorize(self, *args):
        """ memorize(...) -- write memory to manager
        @params args: a sequence of arguments to write """
        self._writer_checkpoint += 1
        args = list(args)
        data = json.dumps([time.time()] + args)
        self._writer.execute(
            'INSERT INTO logs (checkpoint, entry) VALUES (?, ?);',
            (self._writer_checkpoint, data))
        if time.time() - self._writer_flush_time > 2.5:
            self.flush()
            self._writer_flush_time = time.time()
        return

    def get_blob_path(self, epoch, force_save=False):
        """ get_blob_path(epoch, force_save=) -- get epoch blob's path
        @param epoch: epoch num
        @param force_save: if not set then will auto detect if this state is
                           worth saving
        @return path: proposed blob path if it's worth saving or forcefully
                      saved, elsewise None """
        if not force_save and not self._is_valid_epoch_to_save(epoch):
            return None
        path = regularize_path(self._output_path, 'blobs/',
                               'epoch-%d.pth' % epoch)
        return path

    def write_record(self, func, *args, **kwargs):
        """ write_record(func, ..., ...) -- write tensorboard record
        @param func: function name to call
        @note write_record('add_scalar', loss, epoch) is equivalent to
            writer.add_scalar(loss, epoch) """
        if self._tb_writer:
            getattr(self._tb_writer, func)(*args, **kwargs)
        getattr(self._tb_writer_co, func)(*args, **kwargs)
        self.memorize('__main__', 'write-record', func, args, kwargs)
        return

    def recall_filter(self, history):
        """ recall_filter(history) -- filter recall history """
        return

    def recall(self, cnt, timestamp, *args):
        """ recall(cnt, timestamp, ...) -- execute history traceback """
        if cnt == -1:
            return
        if args[0] == 'init':
            if args[1] != self._module_version:
                raise RuntimeError('mismatch HikarI module version')
        elif args[0] == 'write-record':
            self.write_record(args[1], *args[2], **args[3])
        elif args[0] == 'pure-memory':
            pass
        return

    def hook_recall(self, tairitsu, history_id='__main__'):
        """ hook_recall(tairitsu, history_id) -- attach self to recaller """
        tairitsu.register_handler('__main__', self)
        return

    def flush(self):
        """ flush() -- apply memories to disk """
        self._writer_base.commit()
        if self._tb_writer:
            self._tb_writer.flush()
        self._tb_writer_co.flush()
        return

    def close(self):
        """ close() -- close memory manager """
        self.memorize('__main__', 'pure-memory')
        self.flush()
        self._writer_base.close()
        if self._tb_writer:
            self._tb_writer.close()
        self._tb_writer_co.close()
        return
    pass


class TairitsU:
    """ TairitsU -- Experiment history reader and replay manager.
    A harmony of light awaits you in a lost world of musical conflict. """

    def __init__(self, history_path):
        """ TairitsU(history_path):
        @param history_path: memory input path """
        self._input_path = history_path
        self._handlers = {}
        # load cache
        reader_path = regularize_path(self._input_path, 'history.db')
        reader_base = sqlite3.connect(reader_path)
        reader = reader_base.cursor()
        cache = list(reader.execute('SELECT * FROM logs;'))
        self._cache = []
        for _, value in cache:
            timestamp, *value = json.loads(value)
            self._cache.append((timestamp, value))
        reader_base.close()
        return

    def __getitem__(self, index):
        """ __getitem__(index) -- self[index]
        @param index: the index-th history record
        @return timestamp: time executed in history
        @return values: entry itself """
        if not isinstance(index, int):
            raise ValueError('index must be int')
        timestamp, res = self._cache[index]
        return timestamp, res

    def __len__(self):
        """ __len__() -- len(self) """
        return len(self._cache)

    def __iter__(self):
        """ __iter__() -- for i in self """
        for cnt in range(0, len(self._cache)):
            timestamp, val = self[cnt]
            yield timestamp, val
            if val[0] == '__main__' and val[1] == 'pure-memory':
                break
        return

    def get_blob_path(self, epoch):
        """ get_blob_path(epoch) -- retrieve the data blob at epoch
        @param epoch: epoch num
        @return path: supposed path of data blob at epoch """
        path = regularize_path(self._input_path, 'blobs/',
                               'epoch-%d.pth' % epoch)
        return path

    def register_handler(self, key, handler):
        """ register_handler(key, handler) -- register entry handler
        @param key: the entry category, e.g. '__main__'
        @param handler: a class implementing methods:
            recall_filter(history): filter recall history before process begin
                format: (cnt, timestamp, [values...])
            recall(cnt, timestamp, ...): process history record
                cnt = -1 marks termination of recall process
            hook_recall(tairitsu, history_id): attach to recaller """
        if key in self._handlers:
            raise ValueError('conflict key "%s"' % key)
        if not hasattr(handler, 'recall') or not callable(handler.recall):
            raise AttributeError('handler unable to handle event')
        self._handlers[key] = handler
        return

    def recall(self, target_func=None):
        """ recall(target_func=None) -- fast-forward history to checkpoint
        @param target_func: terminates recall process if it returns True """
        if not callable(target_func):
            target_func = (lambda x: False)
        # filter entries in range
        history = []
        for i, (timestamp, value) in enumerate(self):
            if target_func(i, timestamp, *value) is True:
                break
            history.append((i, timestamp, value))
        # tell handlers to pre-filter entries
        for handler_id in self._handlers:
            handler = self._handlers[handler_id]
            hist = []
            for i, timestamp, value in history:
                if value[0] == handler_id:
                    hist.append((i, timestamp, value[1:]))
            if hasattr(handler, 'recall_filter'):
                handler.recall_filter(hist)
        # recall memories
        for i, timestamp, value in history:
            if value[0] in self._handlers:
                handler = self._handlers[value[0]]
                args = value[1:]
                handler.recall(i, timestamp, *args)
        # recall process ended
        for handler in self._handlers:
            self._handlers[handler].recall(-1, 0.0)
        return

    def close(self):
        """ close() -- close history recaller """
        return
    pass


class ShirabE(torch.utils.data.Dataset):
    """ ShirabE: High performance dynamic dataset
    Earn +5 Fragments when playing a Conflict Side song """

    def __init__(self, img_dir, device=None, data_transform=None,
                 input_transform=None, output_transform=None, cache=None,
                 rng=None, history=None, history_id='', **kwargs):
        """ ShirabE(img_dir, device=, data_transform=, input_transform=,
                    output_transform=, cache=, rng=, history=, history_id=)
        @param img_dir: path to dataset
        @param device: pytorch device
        @param data_transform: transformations applied to initial data
        @param input_transform: convert data_transform-ed data to input (X)
        @param output_transform: convert data_transform-ed data to output (Y)
        @param cache: configurations to ShirabE:
            enabled: random cache mode on, default to off
            ttl: any data would be only processed once, default to 8
            size: size of cache buffer, default to 64
        @param rng: random number generator manager, i.e. InoRI
        @param history: history manager, e.g. HikarI
        @param history_id: string id to write history as """
        super(ShirabE, self).__init__(**kwargs)
        # set configuration
        self._module_version = 'ShirabE/20191116.a'
        self._device = device if device else torch.device('cpu')
        if not isinstance(cache, dict):
            cache = {'enabled': False}
        self._cached = True if cache.get('enabled', False) is True else False
        self._ttl = cache.get('ttl', 8)
        self._cache_size = cache.get('size', 64)
        self._path = regularize_path(img_dir)
        # transforms
        self._tsf_data = data_transform
        self._tsf_in = input_transform
        self._tsf_out = output_transform
        # load file directory
        self._files = list(sorted(os.listdir(self._path)))
        self._cache_size = min(self._cache_size, len(self._files))
        self._cc_cnt = [0 for i in self._files]
        self._cc_pool = [None for i in self._files]
        self._cc_seed = ['' for i in self._files]
        self._cc_buffer = []
        # write history
        self._history = history
        self._history_id = history_id
        self._memorize('init', self._module_version, self._cached,
                       self._ttl, self._cache_size,
                       '; '.join(sorted(self._files)))
        # recall manager
        self._rng = rng
        return

    def _memorize(self, *args):
        """ _memorize(...) -- write history """
        if self._history:
            args = [self._history_id] + list(args)
            self._history.memorize(*args)
        return

    def recall_filter(self, history):
        """ recall_filter(history) -- filter recall history """
        return

    def recall(self, cnt, timestamp, *args):
        """ recall(cnt, timestamp, ...) -- execute history traceback """
        if cnt == -1:
            for idx in self._cc_buffer:
                if self._cc_pool[idx] is not None or self._cc_seed[idx] == '':
                    continue
                rng_state = self._rng.get_state()
                self._rng.seed(seed=self._cc_seed[idx])
                self._cc_pool[idx] = self._load_object(idx)
                self._rng.set_state(rng_state)
            return
        if args[0] == 'init':
            if args[1] != self._module_version:
                raise RuntimeError('module version mismatch: %s (%s)' %
                                   (args[1], self._module_version))
            if args[2] != self._cached:
                raise RuntimeError('cache flag mismatch: %s (%s)' %
                                   (args[2], str(self._cached)))
            if args[3] != self._ttl:
                raise RuntimeError('ttl mismatch: %s (%s)' %
                                   (args[3], str(self._ttl)))
            if args[4] != self._cache_size:
                raise RuntimeError('cache size mismatch: %s (%s)' %
                                   (args[4], str(self._cache_size)))
            if args[5] != '; '.join(sorted(self._files)):
                raise RuntimeError('different dataset')
        elif args[0] == 'cache-write':
            ridx, ccnt = args[1], args[2]
            if ccnt == -1:
                self._cc_buffer.remove(ridx)
                self._cc_cnt[ridx] = 0
                self._cc_pool[ridx] = None
                self._cc_seed[ridx] = ''
            else:
                if ridx not in self._cc_buffer:
                    self._cc_buffer.append(ridx)
                self._cc_cnt[ridx] = ccnt
                self._cc_pool[ridx] = None
            self._memorize('cache-write', ridx, ccnt)
        elif args[0] == 'cache-read':
            index, ridx, seed = args[1], args[2], args[3]
            if seed != '':
                self._cc_seed[ridx] = seed
            self._memorize('cache-read', index, ridx, seed)
        return

    def hook_recall(self, tairitsu, history_id):
        """ hook_recall(tairitsu, history_id) -- attach self to recaller """
        tairitsu.register_handler(history_id, self)
        return

    def _load_object(self, index):
        """ _load_object(index) -- render object with id index
        @param index: the # of image to load
        @return x: Tensor, model input (X)
        @return y: Tensor, model output, i.e. ground truth (Y) """
        img = PIL.Image.open(regularize_path(self._path, self._files[index]))
        img = img.convert('YCbCr')
        # do transforms
        if self._tsf_data:
            img = self._tsf_data(img)
        img_x = self._tsf_in(img) if self._tsf_in else img
        img_y = self._tsf_out(img) if self._tsf_out else img
        # send to tensor
        arr_x = numpy.array(img_x).transpose(2, 0, 1) / 255.0
        arr_x = torch.Tensor(arr_x).to(self._device)
        arr_y = numpy.array(img_y).transpose(2, 0, 1) / 255.0
        arr_y = torch.Tensor(arr_y).to(self._device)
        # return values
        return arr_x, arr_y

    def __getitem__(self, index):
        """ __getitem__(index) -- self[index]
        @param index: real idx in files if not cached, otherwise the idx-th
                      element in cache buffer
        @return arr_x: Tensor of the model input
        @return arr_y: Tensor of the model output """
        if not isinstance(index, int):
            raise AttributeError('index must be int')
        if self._cached:
            # fill buffer with objects
            while len(self._cc_buffer) < self._cache_size:
                ridx = -1
                while ridx < 0 or ridx in self._cc_buffer:
                    ridx = random.randrange(0, len(self._files))
                # save to cache pool
                self._cc_cnt[ridx] = random.randrange(0, self._ttl)
                self._cc_pool[ridx] = None
                self._cc_buffer.append(ridx)
                # write history
                self._memorize('cache-write', ridx, self._cc_cnt[ridx])
            if not 0 <= index < self._cache_size:
                raise AttributeError('index out of range')
            ridx = self._cc_buffer[index]
            seed = ''
            if not self._cc_pool[ridx]:
                seed = self._rng.seed()
                self._cc_pool[ridx] = self._load_object(ridx)
                self._cc_seed[ridx] = seed
            arr_x, arr_y = self._cc_pool[ridx]
            self._cc_cnt[ridx] += 1
            self._memorize('cache-write', ridx, self._cc_cnt[ridx])
            if self._cc_cnt[ridx] >= self._ttl:
                self._cc_cnt[ridx] = 0
                self._cc_pool[ridx] = None
                self._cc_buffer.remove(ridx)
                self._memorize('cache-write', ridx, -1)
            self._memorize('cache-read', index, ridx, seed)
        else:
            if not 0 <= index < len(self._files):
                raise AttributeError('index out of range')
            arr_x, arr_y = self._load_object(index)
        return arr_x, arr_y

    def __len__(self):
        """ __len__() -- len(self):
        @return size of cache buffer if cache is enabled, otherwise the size
                of dataset files """
        if self._cached:
            return self._cache_size
        return len(self._files)
    pass


class KuroyuKI:
    """ KuroyuKI: model trainer """

    def __init__(self, model=None, device=None, input_shape=None,
                 upscale_factor=2, dataset_path='./', rng=None,
                 history=None, history_id='', **kwargs):
        """ KuroyuKI(model=, device=, input_shape=, upscale_factor=,
                     dataset_path=, rng=, history=, history_id=)
        @param model: the model to train
        @param device: pytorch device
        @param input_shape: expected model input shape, e.g. (48, 48)
        @param upscale_factor: SR model scales ... times, e.g. 2
        @param dataset_path: path containing both train and test datasets
        @param rng: random number generator manager, i.e. InoRI
        @param history: history record manager, i.e. HikarI,
        @param history_id: string id to write into history as """
        self._module_version = 'KuroyuKI/SR/20191116.a'
        # initialize model
        self._device = device if device else torch.device('cpu')
        self._model = model.to(self._device)
        self._model.eval()
        self._model.train()
        x = torch.zeros((1, 3, ) + input_shape).to(self._device)
        output_shape = tuple(i * upscale_factor for i in input_shape)
        with torch.no_grad():
            y = self._model(x)
            if tuple(y.shape) != (1, 3, ) + output_shape:
                raise RuntimeError('output shape mismatch: %s (should be %s)' %
                                   (tuple(repr(y.shape)),
                                    repr((1, 3, ) + output_shape)))
        self._model_input_shape = input_shape
        self._model_output_shape = output_shape
        self._model_upscale_factor = upscale_factor
        # create data loader
        self._ds_train = ShirabE(
            regularize_path(dataset_path, 'train'),
            device=self._device,
            data_transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(output_shape),
                torchvision.transforms.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.12, hue=0.1),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
            ]),
            input_transform=torchvision.transforms.Resize(input_shape),
            output_transform=None,
            cache={'enabled': True, 'ttl': 8, 'size': 64},
            rng=rng,
            history=history,
            history_id=history_id + '/dataset-train')
        self._ds_test = ShirabE(
            regularize_path(dataset_path, 'test'),
            device=self._device,
            data_transform=torchvision.transforms.CenterCrop(output_shape),
            input_transform=torchvision.transforms.Resize(input_shape),
            output_transform=None,
            cache={'enabled': False},
            rng=rng,
            history=history,
            history_id=history_id + '/dataset-test')
        self._ld_train = torch.utils.data.DataLoader(
            self._ds_train, batch_size=2, num_workers=0, shuffle=False)
        self._ld_test = torch.utils.data.DataLoader(
            self._ds_test, batch_size=2, num_workers=0, shuffle=False)
        # create optimizer
        self._optim = torch.optim.Adam(self._model.parameters(), lr=1e-4,
                                       betas=(0.9, 0.999), weight_decay=1e-3)
        # define loss functions
        self._loss = torch.nn.L1Loss(reduction='mean')
        # self._loss_disp = (lambda x, y: -10 * torch.log10(
        #                    torch.nn.functional.mse_loss(x, y)))
        self._loss_disp = (lambda x, y: -10 * torch.log10(
                           ((x - y) ** 2).sum(dim=(1, 2, 3)) / y.shape[1] /
                           y.shape[2] / y.shape[3]).sum() / y.shape[0])
        # variables
        self.epochs = 0
        self.backprops = 0
        # write history
        self._history = history
        self._history_id = history_id
        self._memorize('init', self._module_version,
                       str(tuple(self._model_input_shape)),
                       self._model_upscale_factor)
        self._rng = rng
        # recall manager (if viable)
        self._recall = None
        self._recall_list = []
        pass

    def _memorize(self, *args):
        """ _memorize() -- write history """
        if self._history:
            args = [self._history_id] + list(args)
            self._history.memorize(*args)
        return

    def recall_filter(self, history):
        """ recall_filter(history) -- filter recall history """
        for cnt, timestamp, value in history:
            if value[0] == 'save-model':
                self._recall_list.clear()
            self._recall_list.append(cnt)
        return

    def recall(self, cnt, timestamp, *args):
        """ recall(cnt, timestamp, ...) -- execute history traceback """
        if cnt == -1:
            return
        if args[0] == 'epoch-begin':
            epoch = args[1]
            debug_log('epoch %d: from recall' % epoch)
        if cnt not in self._recall_list:
            return
        if args[0] == 'init':
            if args[1] != self._module_version:
                raise RuntimeError('module version mismatch: %s (%s)' %
                                   (args[1], self._module_version))
            if args[2] != str(tuple(self._model_input_shape)):
                raise RuntimeError('model input shape mismatch')
            if args[3] != self._model_upscale_factor:
                raise RuntimeError('model upscale factor mismatch')
        elif args[0] == 'epoch-begin':
            epoch, seed = args[1], args[2]
            self._rng.seed(seed=seed)
        elif args[0] == 'save-model':
            self.epochs = int(args[1])
            checkpoint = torch.load(self._recall.get_blob_path(self.epochs))
            self._model.load_state_dict(checkpoint['model'])
            self._optim.load_state_dict(checkpoint['optim'])
            self._model.eval()
            self._model.train()
        elif args[0] == 'epoch-end':
            pass
        return

    def hook_recall(self, tairitsu, history_id):
        """ hook_recall(tairitsu, history_id) -- attach self to recaller """
        self._recall = tairitsu
        tairitsu.register_handler(history_id, self)
        self._ds_train.hook_recall(tairitsu, history_id + '/dataset-train')
        self._ds_test.hook_recall(tairitsu, history_id + '/dataset-test')
        return

    def save_model(self):
        """ save_model() -- save current model """
        if not self._history:
            return
        save_path = self._history.get_blob_path(self.epochs)
        if save_path is not None:
            torch.save({
                'model': self._model.state_dict(),
                'optim': self._optim.state_dict(),
            }, save_path)
            self._memorize('save-model', self.epochs)
        return

    def evaluate_performance(self):
        """ evaluate_performance() -- evaluate model performance on test set
        @return test_l: test loss
        @return test_ld: displayed test acc (e.g. PSNR) """
        test_l_s = torch.zeros(1).to(self._device)
        test_ld_s = torch.zeros(1).to(self._device)
        test_tm_begin = time.time()
        n = torch.zeros(1).to(self._device)
        for x, y in self._ld_test:
            with torch.no_grad():
                y_hat = self._model(x)
                ls = self._loss(y_hat, y)
                test_l_s += ls * y.shape[0]
                ld = self._loss_disp(y_hat, y)
                test_ld_s += ld * y.shape[0]
                n += y.shape[0]
                flush_cache(self._device)
        test_l = float(test_l_s / n)
        test_ld = float(test_ld_s / n)
        test_tm = time.time() - test_tm_begin
        return test_l, test_ld, test_tm

    def epoch(self, force_seed=None):
        """ epoch(force_seed=) -- run epoch
        @param force_seed: set seed if this epoch requires static seed """
        train_l_s = torch.zeros(1).to(self._device)
        train_ld_s = torch.zeros(1).to(self._device)
        train_tm_begin = time.time()
        n = torch.zeros(1).to(self._device)
        if self._rng:
            if force_seed:
                self._memorize('epoch-begin', self.epochs + 1, force_seed)
                self._rng.seed(seed=force_seed)
            else:
                seed = self._rng.seed()
                self._memorize('epoch-begin', self.epochs + 1, seed)
        for x, y in self._ld_train:
            y_hat = self._model(x)
            ls = self._loss(y_hat, y)
            train_l_s += ls * y.shape[0]
            ld = self._loss_disp(y_hat, y)
            train_ld_s += ld * y.shape[0]
            self._optim.zero_grad()
            ls.backward()
            self._optim.step()
            n += y.shape[0]
            self.backprops += 1
            flush_cache(self._device)
        self.epochs += 1
        train_l = float(train_l_s / n)
        train_ld = float(train_ld_s / n)
        train_tm = time.time() - train_tm_begin
        test_l, test_ld, test_tm = float('nan'), float('nan'), float('nan')
        if self.epochs % 10 == 0:
            test_l, test_ld, test_tm = self.evaluate_performance()
        # write history
        if self._history:
            self._history.write_record(
                'add_scalar', 'Loss/train', train_l, self.epochs)
            self._history.write_record(
                'add_scalar', 'Performance/train', train_ld, self.epochs)
            self._history.write_record(
                'add_scalar', 'Time/train', train_tm, self.epochs)
            if test_l >= 0 or test_l <= 0:  # not nan
                self._history.write_record(
                    'add_scalar', 'Loss/test', test_l, self.epochs)
                self._history.write_record(
                    'add_scalar', 'Performance/test', test_ld, self.epochs)
                self._history.write_record(
                    'add_scalar', 'Time/test', test_tm, self.epochs)
            self.save_model()
        # done
        self._memorize('epoch-end', self.epochs)
        debug_log('epoch %d: train %.4f, %.2f (%.3fs) test %.4f %.2f (%.3fs)' %
                  (self.epochs, train_l, train_ld, train_tm, test_l, test_ld,
                   test_tm))
        return
    pass


def train(model_args, dataset_path, input_shape, upscale_factor, epochs,
          history=None, force_seed=None, recall=None, start_from_epoch=1,
          tb_log_path=None):
    """ train(model_args, dataset_path, input_shape, upscale_factor, epochs,
              history=, force_seed=, recall=, start_from_epoch,
              tb_log_path=) -- train model
    @param model_args: pass (class, args, kwargs) as arguments, e.g.
        (WaifuModel, [arg1, arg2], dict(x=1, y=2)) is equivalent to creating
        WaifuModel(arg1, arg2, x=1, y=2)
    @param dataset_path: the path to directory containing train and test sets
    @param input_shape: model input shape, e.g. (48, 48)
    @param upscale_factor: superresolution upscale factor, e.g. 4
    @param epochs: run until epoch num
    @param history: experiment history output path
    @param force_seed: use this seed as init, if not recalling
    @param recall: recall with experiment history output path
    @param start_from_epoch: recall starting from epoch num, default to 1
    @param tb_log_path: tensorboard log path output """
    device = get_device()
    hikari = HikarI(history, log_path=tb_log_path) if history else None
    inori = InoRI(seed=force_seed, history=hikari, history_id='rng')
    model = model_args[0](*model_args[1], **model_args[2])
    kuroyuki = KuroyuKI(model=model, device=device, input_shape=input_shape,
                        upscale_factor=upscale_factor,
                        dataset_path=dataset_path, rng=inori, history=hikari,
                        history_id='trainer')
    # recall previous epochs
    if recall:
        tairitsu = TairitsU(recall)
        if hikari:
            hikari.hook_recall(tairitsu)
        inori.hook_recall(tairitsu, 'rng')
        kuroyuki.hook_recall(tairitsu, 'trainer')
        # forward epoch count to model savepoint
        b_ep = max(start_from_epoch - 1, 0)
        while b_ep > 0 and not os.path.exists(tairitsu.get_blob_path(b_ep)):
            b_ep -= 1
        # trigger recall
        judge = (lambda _i, _t, *v: v[0] == 'trainer' and v[1] == 'epoch-begin'
                 and v[2] >= b_ep + 1)
        tairitsu.recall(target_func=judge)
    else:
        kuroyuki.save_model()
    # run until finish
    while kuroyuki.epochs < epochs:
        kuroyuki.epoch()
    # close handles
    if recall:
        tairitsu.close()
    if hikari:
        hikari.close()
    return
