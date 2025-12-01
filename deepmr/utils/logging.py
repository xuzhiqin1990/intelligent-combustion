import logging

# 可以追加输入的日志
# 调用方式为 my_logger = Log('./my_logger.log')
# 若不存在则创建，若存在则追加写入
class Log:
    def __init__(self, file_name, mode = 'a'):
        # 第一步，创建一个logger
        self.logger = logging.getLogger(file_name)  # file_name为多个logger的区分唯一性
        self.logger.setLevel(logging.DEBUG)  # Log等级总开关
        # 如果已经有handler，则用追加模式，否则直接覆盖
        # mode = 'a' if self.logger.handlers else 'w'
        # 第二步，创建handler，用于写入日志文件和屏幕输出
        fmt = "%(asctime)s - %(levelname)s: %(message)s"
        formatter = logging.Formatter(fmt)
        # 文件输出
        fh = logging.FileHandler(file_name, mode=mode)
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        fh.setFormatter(formatter)
        # # # 往屏幕上输出
        # sh = logging.StreamHandler()
        # sh.setFormatter(formatter)  # 设置屏幕上显示的格式
        # sh.setLevel(logging.DEBUG)
        # 先清空handler, 再添加
        self.logger.handlers = []
        self.logger.addHandler(fh)
        # self.logger.addHandler(sh)

    def info(self, message):
        self.logger.info(message)


def setup_logger(log_file, level=logging.INFO):
    l = logging.getLogger(log_file[:-4])
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fileHandler)