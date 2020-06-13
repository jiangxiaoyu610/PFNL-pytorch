from local_common import *


class Args:
    """
    各种参数集合
    """
    def __init__(self):
        self.num_frames = 7
        self.scale = 4
        self.in_size = 32
        self.ground_truth_size = self.in_size * self.scale

        self.batch_size = 1  # 原始参数为 16
        self.eval_batch_size = 1 # 原始参数为 4
        self.eval_in_size = [128, 240]

        self.n_filters = 64  # 此参数值针对 conv0、conv1、conv10、conv2
        self.k_size = 3  # kernel size, 此参数只针对 conv1、conv2
        self.strides = 1  # 此参数针对所有卷积层
        self.n_block = 20
        self.activate = torch.nn.LeakyReLU

        self.decay_step = 1.2e5
        self.max_step = int(1.5e5 + 1)
        self.learning_rate = 1e-3
        self.end_learning_rate = 1e-4

        self.reload = False
        self.train_files_list = './data/filelist_train.txt'
        self.eval_files_list = './data/filelist_val.txt'

        self.log_dir = './logs/'
        self.save_dir = './checkpoint/'
        self.checkpoint_file = None
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
