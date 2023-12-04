import wandb

from utils.logs import log_func


def print_log(dict_log):
    str_print = ""
    for key, value in dict_log.items():
        str_print += f'{key}: {round(value, 8)}; '
    log_func(str_print, level='train_model')


def wandb_log(dict_log):
    print_log(dict_log)
    dict_log = dict([(key, item) for key,
                                     item in dict_log.items() if 'epoch' not in key])
    wandb.log(dict_log)
