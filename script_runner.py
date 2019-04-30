# -*- coding: utf-8 -*-

import os

from Zeras.vocab import Vocab
from model_settings import ModelSettings

import argparse

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval', 'predict', 'convert'],
                        default = 'train', help = 'run mode')
    #
    parser.add_argument('--note', type=str, default = 'note_something',
                        help = 'make some useful notes')
    parser.add_argument('--debug', type=int, default = 0,
                        help = 'debug or not (using debug data or not)')
    parser.add_argument('--gpu', type=str, default = '0',
                        help = 'specify gpu device')
    #
    parser.add_argument('--ckpt_loading', choices=['best', 'latest'],
                        default = 'best', help='lastest ckpt or best')
    #
    parser.add_argument('--task', type=str, help = 'specify task',
                        default = 'copy')
    parser.add_argument('--settings', type=str, help='settings file',
                        default = None)
    parser.add_argument('--model_tag', type=str, help='model_tag',
                        default = 'transformer')
    #    
    return parser.parse_args()

#  
if __name__ == '__main__':
    
    args = parse_args()
    run_mode = args.mode
    #
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #
    # task & model
    task = args.task
    settings_file = args.settings
    if task == "copy":
        import task_copy_model_utils as model_utils
        if settings_file is None:
            settings_file = "./task_copy_settings.json"
        #
    #
    model_tag = args.model_tag
    if model_tag.startswith('transformer'):
        from model_graph_transformer import ModelGraph    
    #
    # settings
    settings = ModelSettings()
    settings.load_from_json_file(settings_file)
    settings.gpu_available = args.gpu
    settings.model_tag = model_tag
    #    
    if run_mode == 'predict':
        settings.is_train = False
    else:
        settings.is_train = True
    #
    settings.check_settings()
    settings.create_or_reset_log_file()
    settings.logger.info('running with args : {}'.format(args))
    settings.logger.info(settings.trans_info_to_dict())
    settings.save_to_json_file("./temp_settings.json")
    #
    # vocab
    vocab = Vocab()    
    vocab.add_tokens_from_file(settings.tokens_file)
    vocab.load_pretrained_embeddings(settings.emb_file)
    vocab.emb_dim = settings.emb_dim
    #
    # model & vocab
    settings.model_graph = ModelGraph
    settings.vocab = vocab
    #
    # run
    if run_mode == 'train':
        model_utils.do_train_and_valid(settings, args)
    elif run_mode == 'eval':
        model_utils.do_eval(settings, args)
    elif run_mode == 'predict':
        model_utils.do_predict(settings, args)
    elif run_mode == 'convert':
        model_utils.do_convert(settings, args)
    else:
        print('NOT supported mode. supported modes: train, eval, convert and predict.')
    #
    
    