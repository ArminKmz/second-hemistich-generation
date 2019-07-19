import random
import numpy as np

class data_loader:
    def __init__(self, path):
        self.path = path
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
            self.max_size  = 0
            self.all_chars_set = {'_PAD_', '_BOM_', '_EOM_'}
            for line in lines:
                line = line.rstrip('\n')
                [first_h , second_h] = line.split(',')
                first_h = first_h.rstrip(' ').lstrip(' ')
                second_h = second_h.rstrip(' ').lstrip(' ')
                for c in first_h + second_h:
                    self.all_chars_set.add(c)
                self.max_size = max(self.max_size, len(first_h))
                self.max_size = max(self.max_size, len(second_h))
            self.all_chars_dic = {c : idx for idx, c in enumerate(self.all_chars_set)}
            self.n_chars = len(self.all_chars_dic)
            self.n_examples = len(lines)
            self.max_size += 2

    def get_data(self):
        X = np.zeros((self.n_examples, self.max_size))
        Y = np.zeros((self.n_examples, self.max_size))
        with open(self.path, encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.rstrip('\n')
                [first_h , second_h] = line.split(',')
                first_h = first_h.rstrip(' ').lstrip(' ')
                second_h = second_h.rstrip(' ').lstrip(' ')
                X[i, :] = self.line2tensor(first_h, True)
                Y[i, :] = self.line2tensor(second_h, False)
        return X, Y

    def char2idx(self, c):
        return self.all_chars_dic[c]

    def idx2char(self, idx):
        for k, v in self.all_chars_dic.items():
            if v == idx:
                return k
        raise Exception('There is no', idx, ' in all_chars_dic')

    def tensor2line(self, tensor, tags=True):
        if tags:
            return ''.join(self.idx2char(tensor[i]) for i in range(tensor.shape[0]))
        else:
            return ''.join(self.idx2char(tensor[i]) for i in range(tensor.shape[0])
                            if self.idx2char(tensor[i]) not in ['_BOM_', '_EOM_', '_PAD_'])

    def line2tensor(self, line, first_flag):
        # add _BOM_ at the begining of line and
        # _EOM_ at the end of line and _PAD_ based
        # on first_flag
        sz = self.max_size
        tensor = np.zeros(sz)
        # pad from begining if it's first hemistich
        if first_flag:
            for i in range(0, sz-len(line)-2):
                tensor[i] = self.char2idx('_PAD_')
            tensor[sz-len(line)-2]  = self.char2idx('_BOM_')
            offset = sz - len(line) - 1
            tensor[-1] = self.char2idx('_EOM_')
        else:
            tensor[0] = self.char2idx('_BOM_')
            offset = 1
        for i, c in enumerate(line):
            tensor[i+offset] = self.char2idx(c)
            idx = self.char2idx(c)
        # pad at the end if it's second hemistich
        if not first_flag:
            tensor[len(line)+offset] = self.char2idx('_EOM_')
            offset += 1
            for i in range(len(line)+offset, sz):
                tensor[i] = self.char2idx('_PAD_')
        return tensor
