import os
import logging


def getLogger(name, path, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    os.makedirs(path, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s: %(message)s")
    log_file = f'{path}train.log'
    with open(log_file, 'w'):
        pass
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger
