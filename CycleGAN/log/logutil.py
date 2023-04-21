import logging

# Ref: https://zhuanlan.zhihu.com/p/166671955

log_level_map = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critival': logging.CRITICAL
}

class Logger:
    def __init__(self, log_name='out.log', log_id='', log_dir='./'):
        # log_name: log名字
        # log_id: 副名

        self.logger_name_dict = {}
        self.log_dir = log_dir
        self.log_name = log_name

        if log_id is None:
            self.main_name = 'main'
        else:
            self.main_name = str(log_id)

        self.logger_name_dict[self.main_name] = []
        self.logger = logging.getLogger(self.main_name)
        self.logger.setLevel(logging.INFO)

        fh, sh = self.log_format(log_name=self.log_name)
        self.fh = fh  # file handler
        self.sh = sh  # stream handler
        if self.logger.handlers:
            self.logger.handlers = []

        # 添加 handler
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

        
    def log_format(self, level='debug', file_level='debug', log_name='out.log'):
        """

        :param level: print log level
        :param file_level: log file log level
        :param log_path: log file path
        :return:
        """

        #self.log_dir = log_dir
        #self.log_dir = './'
        logname =  self.log_dir  +'/'+ log_name
        fh = logging.FileHandler(logname, mode='a', encoding='utf-8')
        fh.setLevel(log_level_map[file_level])

        sh = logging.StreamHandler()
        sh.setLevel(log_level_map[level])

        # 定义handler的输出个数
        formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s: %(message)s',
                                      datefmt="%Y/%m/%d %H:%M:%S")
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)

        return fh,sh

    def set_sub_logger(self, name):
        if name not in self.logger_name_dict[self.main_name]:
            new_logger = logging.getLogger(self.main_name + "."+name)
            self.logger_name_dict[self.main_name].append(new_logger)
        else:
            new_logger = logging.getLogger(self.main_name + "."+name)

        return new_logger

    
    def remove_main_logger(self, name):
        if name in self.logger_name_dict.keys():
            for i in self.logger.handlers:
                self.logger.removeHandler(i )


            self.logger_name_dict.pop(self.main_name, 0)


