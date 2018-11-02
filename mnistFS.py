import  tensorflow as tf
from    tensorflow.examples.tutorials.mnist import input_data
import  numpy as np
from    multiprocessing import Pool
from    matplotlib import pyplot as plt

from    PIL import Image
from    torchvision import transforms

class MnistFS:

    def __init__(self, path, mode='train', n_way = 5, k_spt=1, k_qry=15):
        """
        :param path:
        :param mode: train/val/test
        :param n_way: n-way categories
        :param k_spt: k-shot per category
        :param k_qry: k-shot query images per category
        """

        mnist = input_data.read_data_sets(path, one_hot=False, validation_size=0)
        data1 = mnist.train.next_batch(60000)
        data2 = mnist.test.next_batch(10000)

        x = np.concatenate([data1[0], data2[0]], axis=0)
        label = np.concatenate([data1[1], data2[1]], axis=0)
        print(x.shape, label.shape)

        # every element is a list including all images belonging to same category
        # TODO: CAN NOT write as [[]]*10, since data[0] and data[1] will use same reference
        data = [[], [], [], [], [], [], [], [], [], []]
        for i, j in zip(x, label):
            data[j].append(i)
        data = list(map(lambda x:np.array(x), data))

        num_len = list(map(lambda x: x.shape[0], data))
        print(num_len)

        if mode == 'train':
            mode_numbers =[0, 1, 2, 3, 4]
        else:
            mode_numbers =[5, 6, 7, 8, 9]

        assert n_way >= len(mode_numbers)


        # convert list to dict
        self.data = dict()
        for i in mode_numbers:
            self.data[str(i)] = data[i]

        self.n_way = n_way
        self.k_spt = k_spt
        self.k_qry = k_qry
        self.mode_numbers = mode_numbers


    def union_shuffle(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def get(self, dummy=0):
        """

        :param dummy: dummy parameters, to support multiprocessing.pool.map
        :return:
        """
        # sample n classes
        sampled_numbers = np.random.choice(self.mode_numbers, size=self.n_way, replace=False)
        # convert int to str
        sampled_numbers = map(lambda x:str(x), sampled_numbers)
        # save x and y
        spt_x, spt_y, qry_x, qry_y = [], [], [], []

        # for each label
        for l in sampled_numbers:
            # sample idx first since random.choice only support 1-d data
            sampled_idx = np.random.choice(len(self.data[l]), size=(self.k_qry + self.k_spt), replace=False)
            # [15+5, 784]
            sampled_x =  self.data[l][sampled_idx]

            # print(sampled_x.shape)
            # list append with np.aray(15+1, 784)
            spt_x.append(sampled_x[:self.k_spt])
            qry_x.append(sampled_x[self.k_spt:])

            # [...., ..., ...]
            spt_y.extend( [int(l)]*self.k_spt )
            qry_y.extend( [int(l)]*self.k_qry )
        spt_y, qry_y = np.array(spt_y), np.array(qry_y)

        # list of [n*(k_spt+k_qry)] to [b, img]
        spt_x = np.concatenate(spt_x, axis=0)
        qry_x = np.concatenate(qry_x, axis=0)

        # print(spt_x.shape, spt_y.shape)


        spt_x, spt_y = self.union_shuffle(spt_x, spt_y)
        qry_x, qry_y = self.union_shuffle(qry_x, qry_y)


        transform = transforms.Compose([
            # [784] => [28, 28]
            lambda x: x.reshape(28, 28),
            # convert to PIL.Image from numpy
            lambda x: Image.fromarray(np.uint8(x*255)),
            # resize
            transforms.Resize((32, 32)),
            # flatten
            lambda x: np.array(x).reshape(-1)
        ])
        spt_x = list(map(lambda x:transform(x), spt_x))
        spt_x = np.array(spt_x)
        qry_x = list(map(lambda x:transform(x), qry_x))
        qry_x = np.array(qry_x)


        return spt_x, spt_y, qry_x, qry_y

    def get_batch(self, batchsz, use_episode=True):
        """
        fetch data parallel
        :param batchsz:
        :param use_episode: if use episode based training, return [spt_x, spt_y, qry_x, qry_y]
                            Otherwise, it will merge spt and qry sets and return [batchx, batchy]
        :return:
        """

        pool = Pool(processes=batchsz)
        # list of (spt_x, spt_y, qry_x, qry_y)
        batch = pool.map(self.get, [0]*batchsz)
        pool.close()
        pool.join()

        # print(batch)
        # list of (spt_x, spt_y, qry_x, qry_y) tuple
        spt_x, spt_y, qry_x, qry_y = [b[0] for b in batch], [b[1] for b in batch], \
                                     [b[2] for b in batch], [b[3] for b in batch]
        # [b, sprt_x], [b, n_way]
        spt_x, spt_y, qry_x, qry_y = np.stack(spt_x, axis=0), np.stack(spt_y, axis=0),\
                                     np.stack(qry_x, axis=0), np.stack(qry_y, axis=0)
        # (8, 5, 784) (8, 75, 784) (8, 5) (8, 75)
        # => [b, 5, 32, 32, 1], [b, 75, 32, 32, 1]
        spt_x, qry_x = spt_x.reshape(batchsz, -1, 32, 32, 1), qry_x.reshape(batchsz, -1, 32, 32, 1)
        # print(spt_x.shape, qry_x.shape, spt_y.shape, qry_y.shape)

        if use_episode:
            return spt_x, spt_y, qry_x, qry_y
        else:
            # [b, 5, 32, 32, 1] [b, 75, 32, 32, 1]
            batchx = np.concatenate([spt_x, qry_x], axis=1)
            # [b, 5] [b, 75]
            batchy = np.concatenate([spt_y, qry_y], axis=1)
            # shuffle once again
            # TODO: shuffle here!
            # batchx, batchy = self.union_shuffle(batchx, batchy)

            return batchx, batchy

def main():
    import time

    mnist = MnistFS('ae_data/mnist', mode='test')

    row = 5
    column = 16


    fig = plt.figure()

    plt.ion()

    # while True:
    #     _, _, x, y = mnist.get()
    #     for i in range(75):
    #         fig.add_subplot(row, column, i+1)
    #
    #         # print(x[i].shape)
    #         img = x[i].reshape(28, 28)
    #         plt.imshow(img)
    #     plt.pause(0.001)
    #     print(y)
    #
    #     time.sleep(8)


    while True:
        _, _, x, y = mnist.get_batch(8)
        batchidx = np.random.randint(8)
        x, y = x[batchidx], y[batchidx]
        for i in range(75):
            fig.add_subplot(row, column, i+1)

            # print(x[i].shape)
            img = x[i].reshape(32, 32)
            plt.imshow(img)
        plt.pause(0.001)
        print(y)

        time.sleep(8)

if __name__ == '__main__':
    main()