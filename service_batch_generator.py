#-*- coding: utf-8 -*-
import numpy as np

class ServiceBatchGenerator(object):
    """
        Implementation of a random service chain generator

        Attributes:
            state[batchSize, maxServiceLength] -- Generated random service chains
            serviceLength[batchSize] -- Generated array contining services length
    """
    def __init__(self, batchSize, minServiceLength, maxServiceLength, numDescriptors):
        """
        Args:
            batchSize(int) -- Number of service chains to be generated
            minServiceLength(int) -- Minimum service length
            maxServiceLength(int) -- Maximum service length
            numDescriptors(int) -- Number of unique descriptors
        """
        self.batchSize = batchSize
        self.minServiceLength = minServiceLength
        self.maxServiceLength = maxServiceLength
        self.numDescriptors = numDescriptors

        self.serviceLength = np.zeros(self.batchSize,  dtype='int32')
        self.state = np.zeros((self.batchSize, self.maxServiceLength),  dtype='int32')
        self.prob = [0.1, 0.1, 0.1, 0.2, 0.2, 0.05, 0.05]


    def getNewState(self):
        """ Generate new batch of service chain """

        # Clean attributes
        self.serviceLength = np.zeros(self.batchSize,  dtype='int32')
        self.state = np.zeros((self.batchSize, self.maxServiceLength),  dtype='int32')

        # Compute random services
        for batch in range(self.batchSize):
            self.serviceLength[batch] = np.random.randint(self.minServiceLength, self.maxServiceLength+1, dtype='int32')
            """
            生成符合self.prob多项分布的任务
            但是这个函数无法产生随机数 只能直接输出一批样本 
            """
            sample = np.random.multinomial(self.serviceLength[batch], self.prob, size=1)[0]
            # 将多项分布样本编码到state中
            serviceNo = 0
            for i in range(len(sample)):
                for j in range(sample[i]):
                    self.state[batch][serviceNo] = i
                    serviceNo += 1
            # 当前结果是任务类别 从小到大排列的 进行shuffle
            temp = self.state[batch][0:self.serviceLength[batch]]
            np.random.shuffle(temp)
            self.state[batch][0:self.serviceLength[batch]] = temp

            """
            生成均匀分布的任务
            """
            # for i in range(self.serviceLength[batch]):
            #     pktID = np.random.randint(0, self.numDescriptors,  dtype='int32')
            #     self.state[batch][i] = pktID


if __name__ == "__main__":

    # Define generator
    batch_size = 5
    minServiceLength = 2
    maxServiceLength = 6
    numDescriptors = 7

    '''创建一个数据生成器对象'''
    env = ServiceBatchGenerator(batch_size, minServiceLength, maxServiceLength, numDescriptors)
    '''生成新的数据'''
    """
    数据存储在serviceLength、state两个ndarray中
    serviceLength是一个长度为batchSize的一维向量 第i个元素的值表示这个batch中第i组数据（待调度任务链）的有效长度
    state是一个shape[batchSize, maxServiceLength]的二维向量 每i行表示第i组数据（待调度任务连） 第i组数据的有效长
        度由serviceLength[i]决定 元素的值表示任务类别（numDescriptors种） 任务类别与任务的资源需求相关 对应关系在environment中定义 
    """
    env.getNewState()

    print()
