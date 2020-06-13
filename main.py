import re
import time

from PFNL import PFNL
from local_common import *
from dataset import MyDataSet
from argument import Args
from torch.utils.data import DataLoader


def learning_rate_decay(old_lr, global_step, args, power=1):
    """
    利用 tensor flow 中的 polynomial_decay 进行学习率衰减。
    :param old_lr:
    :param global_step:
    :param args:
    :param power:
    :return:
    """
    decay_step = args.decay_step
    global_step = min(global_step, decay_step)

    end_lr = args.end_learning_rate

    decayed_lr = (old_lr - end_lr) * ((1 - global_step / decay_step) ** power) + end_lr

    return decayed_lr


def show_num_of_trainable_params(model):
    """
    展示网络可训练参数的个数。
    :param model:
    :return:
    """
    n_params = 0
    for params in model.parameters():
        if params.requires_grad:
            n_params += params.numel()

    print('Params num of all: {}'.format(n_params))

    return


def get_data_loaders(args):
    """
    生成数据集读取器
    :param args:
    :return:
    """
    data_set = MyDataSet(args)
    data_loaders = iter(DataLoader(data_set, batch_size=args.batch_size, shuffle=True))

    return data_loaders


def load_checkpoint(model, optimizer, args):
    """
    读取 checkpoint
    :param model:
    :param optimizer:
    :param args:
    :return:
    """
    filename = args.checkpoint_file
    if filename is None:
        return
    checkpoint = torch.load(filename)

    last_step = checkpoint['step']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, last_step


def clean_save_dir(save_dir, file_list):
    """
    删除最早的 checkpoint
    :param save_dir:
    :param file_list:
    :return:
    """
    min_step = float('inf')
    for filename in file_list:
        step = re.match('[0-9]*', filename)
        min_step = min(min_step, int(step))

    remove_file = save_dir + 'checkpoint_{}.pkl'.format(min_step)
    os.remove(remove_file)

    return


def save_checkpoint(model, optimizer, step, args, max_to_keep=10):
    """
    储存 checkpoint
    :param model:
    :param optimizer:
    :param step:
    :param args:
    :param max_to_keep:
    :return:
    """
    state_dict = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    save_dir = args.save_dir
    filename = os.path.join(save_dir, 'checkpoint_{}.pkl'.format(step))

    file_list = os.listdir(save_dir)
    if len(file_list) > max_to_keep:
        clean_save_dir(save_dir, file_list)

    torch.save(state_dict, filename)

    return


def load_eval_batch(file_list, kernel, out_h, out_w, args):
    """
    读取验证模型用的 sample batch
    :param file_list:
    :param kernel:
    :param out_h:
    :param out_w:
    :param args:
    :return:
    """
    border = 8  # 没搞明白这个边框的意义
    center = 15  # 也没懂这个center，如果是为了从 n frames 中间向两边取，不应该设为 15 吧
    max_frame = len(file_list)

    batch_input_image, batch_ground_truth = [], []
    for tmp_center in range(center, max_frame, 32):  # 为什么跨度设为 32？
        file_index = np.arange(tmp_center - args.num_frames // 2, tmp_center + args.num_frames // 2 + 1)
        file_index = np.clip(file_index, 0, max_frame - 1)

        input_image, ground_truth = [], []
        for i in file_index:
            image = load_image(file_list[i])  # w * h
            image = TENSOR_TRANS(image)  # 注意：这里会变成 h * w
            image = image[:, border: out_h + border, border: out_w + border] / 255.0

            input_image.append(image)
            ground_truth.append(image)

        input_image = torch.stack(input_image, 0)
        ground_truth = torch.stack(ground_truth, 0)
        input_image = down_sample_with_blur(input_image, kernel, args.scale)

        batch_input_image.append(input_image)
        batch_ground_truth.append(ground_truth)

        if len(batch_input_image) == len(batch_ground_truth) == args.eval_batch_size:
            batch_input_image = torch.stack(batch_input_image, 0)
            batch_ground_truth = torch.stack(batch_ground_truth, 0)
            yield batch_input_image, batch_ground_truth

            batch_input_image, batch_ground_truth = [], []


def write_eval_log(mse_average, psnr_average, step, args):
    """
    写入验证结果日志
    :param mse_average:
    :param psnr_average:
    :param step:
    :param args:
    :return:
    """
    with open(args.log_dir, 'a+') as f:
        mse_average = (mse_average * 1e6).astype(np.int64) / 1e6
        psnr_average = (psnr_average * 1e6).astype(np.int64) / 1e6
        f.write('{ "Iter": {}, "PSNR": {}, "MSE": {} }\n'.format(step, psnr_average.tolist(), mse_average.tolist()))

    return


def mse_reduce_mean(error):
    """
    非通用函数。
    对 error 做 reduce mean，得到 mse。
    :param error:
    :return:
    """
    mse = torch.mean(error, dim=-1)
    mse = torch.mean(mse, dim=-1)
    mse = torch.mean(mse, dim=-1)

    return mse


def eval_model(model, args, step, device):
    """
    在验证集上验证模型
    :param model:
    :param args:
    :param step:
    :param device:
    :return:
    """
    print('Evaluating:')
    model.eval()

    in_h, in_w = args.eval_in_size
    out_h = in_h * args.scale  # 512
    out_w = in_w * args.scale  # 960

    kernel = get_gaussian_filter(13, 1.6)  # 13 and 1.6 for x4 down sample
    file_groups = get_data_set_list(args.eval_files_list)

    mse_accuracy = None
    batch_count = 0

    for file_list in file_groups:
        for batch_input_image, batch_ground_truth in load_eval_batch(file_list, kernel, out_h, out_w, args):
            batch_input_image, batch_ground_truth = batch_input_image.to(device), batch_ground_truth.to(device)

            with torch.set_grad_enabled(False):
                output = model(batch_input_image)
                ground_truth = batch_ground_truth[:, args.num_frames // 2, :, :, :].unsqueeze(0)
                # 等价于 tf.reduce_mean((output-ground_truth) ** 2, axis=[2, 3, 4])
                mse = mse_reduce_mean((output - ground_truth) ** 2)

            if mse_accuracy is None:
                mse_accuracy = mse
            else:
                mse_accuracy = torch.cat([mse_accuracy, mse], 0)

            print('finish batch {} - {}'.format(batch_count, batch_count + args.eval_batch_size))
            batch_count += args.eval_batch_size

    psnr_accuracy = 10 * np.log10(1.0 / mse_accuracy)
    mse_average, psnr_average = torch.mean(mse_accuracy, 0), torch.mean(psnr_accuracy, 0)

    print('Eval PSNR: {}, MSE: {}'.format(psnr_average, mse_average))

    write_eval_log(mse_average, psnr_average, step, args)
    model.train()

    return


def forward_and_backward(input_images, ground_truth, model, optimizer, criterion, step, args):
    """
    进行模型的前向传播和反向更新。
    :param input_images:
    :param ground_truth:
    :param model:
    :param optimizer:
    :param criterion:
    :param step:
    :param args:
    :return:
    """
    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
        output = model(input_images)
        loss = criterion(output, ground_truth)

        loss.backward()
        optimizer.step()

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate_decay(param_group['lr'], step, args)

    return loss


def initial_utils(args):
    """
    初始化训练模型用的各类
    :param args:
    :return:
    """
    model = PFNL(args)
    data_loaders = get_data_loaders(args)

    last_step = -1  # 之后遍历时还要 +1。这样是为了和读取出来的last step保持操作一致
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.reload:
        model, optimizer, last_step = load_checkpoint(model, optimizer, args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model.to(device)

    return model, optimizer, criterion, data_loaders, last_step, device


def train_model():
    """
    训练模型
    :return:
    """
    args = Args()

    # 本代码展示的模型参数数量与原代码不同。
    # 原因是NonLocalBlock中的phi、theta层的参数在原代码中没有被计算进来，
    # 但是在本代码中虽然也没有经过并参与训练，但是被计算了参数数量。
    # 经过对模型参数的仔细对比，与原代码所搭建的模型完全一样。
    model, optimizer, criterion, data_loaders, last_step, device = initial_utils(args)
    show_num_of_trainable_params(model)

    model.train()
    for step in range(last_step + 1, args.max_step):
        input_images, ground_truth = next(data_loaders)
        input_images, ground_truth = input_images.to(device), ground_truth.to(device)

        loss = forward_and_backward(input_images, ground_truth, model, optimizer, criterion, step, args)

        if step % 20 == 0:
            local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print("{} step: {}, loss: {}".format(local_time, step, loss))
        if step % 500 == 0:
            if step != 0:
                save_checkpoint(model, optimizer, step, args)

            eval_model(model, args, step, device)

        if step > 500 and loss > 10:
            print('Model collapsed with loss: {}'.format(loss))
            break

    return


if __name__ == '__main__':
    train_model()
