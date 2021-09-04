import os
import sys
import torch
import logging
from datetime import datetime

# return a fake summarywriter if tensorbaordX is not installed
# tensorbaordX 可视化
try:
    from tensorboardX import SummaryWriter
except ImportError:
    # fake summarywriter
    class SummaryWriter:
        def __init__(self, log_dir=None, comment='', **kwargs):
            print('\n unable to import tensorboardX, log will be recorded by pytorch! \n')
            self.log_dir = log_dir if log_dir is not None else './logs'
            os.makedirs('./logs', exist_ok=True)
            self.logs = {'comment': comment}
            return

        def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
            if tag in self.logs:
                self.logs[tag].append((scalar_value, global_step, walltime))
            else:
                self.logs[tag] = [(scalar_value, global_step, walltime)]
            return

        def close(self):
            timestamp = str(datetime.now()).replace(' ', '_').replace(':', '_')
            torch.save(self.logs, os.path.join(self.log_dir, 'log_%s.pickle' % timestamp))
            return


class EmptySummaryWriter:
    def __init__(self, **kwargs):
        pass

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        pass

    def close(self):
        pass


def create_summary(distributed_rank=0, **kwargs):
    # summary_writer = create_summary(cfg.local_rank, log_dir=cfg.log_dir)
    if distributed_rank > 0:
        return EmptySummaryWriter(**kwargs)
    else:
        return SummaryWriter(**kwargs)


# distributed_rank=cfg.local_rank
def create_logger(distributed_rank=0, save_dir=None):
    # 创建logger日志对象
    logger = logging.getLogger('logger')
    # 设置log等级为debug
    logger.setLevel(logging.DEBUG)
    # 设置日志文件名
    filename = "log_%s.txt" % (datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    # don't log results for the non-master process
    # 不记录非主进程的结果
    if distributed_rank > 0:
        return logger
    # StreamHandler 可以向 sys.stdout 输出信息
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    # formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

    # %(asctime)s 字符串形式的当前时间。默认格式是 “2003-07-08 16:49:45,896”。逗号后面的是毫秒
    # %(message)s 用户输出的消息
    formatter = logging.Formatter("%(message)s [%(asctime)s]")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir is not None:
        # FileHandler 日志输出到文件
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# 储存器
class Saver:
    def __init__(self, distributed_rank, save_dir):
        # distributed_rank = cfg.local_rank
        self.distributed_rank = distributed_rank
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        return

    def save(self, obj, save_name):
        if self.distributed_rank == 0:
            # 以.t7格式保存模型
            torch.save(obj, os.path.join(self.save_dir, save_name + '.t7'))
            return 'checkpoint saved in %s !' % os.path.join(self.save_dir, save_name)
        else:
            return ''


# 创建储存器
def create_saver(distributed_rank, save_dir):
    return Saver(distributed_rank, save_dir)


# 禁用打印
class DisablePrint:
    def __init__(self, local_rank=0):
        self.local_rank = local_rank

    # 兼容with语句，上下文管理器
    # with开始运行时调用__enter__方法
    def __enter__(self):
        if self.local_rank != 0:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        else:
            pass

    # with结束后，调用__exit__
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.local_rank != 0:
            sys.stdout.close()
            sys.stdout = self._original_stdout
        else:
            pass
