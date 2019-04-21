# -*- coding: utf-8 -*-

import os

from Zeras.vocab import Vocab

from task_copy_model_settings import ModelSettings
import task_copy_model_utils as model_utils

import argparse


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('task_copy')
    parser.add_argument('--mode', choices=['train', 'eval', 'predict', 'debug'],
                        default = 'train', help = 'run mode')
    #
    parser.add_argument('--note', type=str, default = 'note_something',
                        help = 'make some useful notes')
    parser.add_argument('--debug', type=int, default = 0,
                        help = 'debug or not (using debug data or not)')
    parser.add_argument('--gpu', type=str, default = '0',
                        help = 'specify gpu device')
    #
    parser.add_argument('--max_batches_eval', type=int, default = 20,
                        help = 'specify how many batches go through eval')
    #
    model_related = parser.add_argument_group('model related settings')
    model_related.add_argument('--model_tag', type=str,
                               default = 'transformer', help='model_tag')
    model_related.add_argument('--base_dir', type=str,
                               default = 'task_copy_result',
                               help='base directory for saving models')
    #
    vocab_related = parser.add_argument_group('vocab related settings')
    vocab_related.add_argument('--emb_file', type=str, default = None,
                               help='pretrained embeddings file')
    vocab_related.add_argument('--tokens_file', type=str,
                               default = './vocab/vocab_tokens.txt',
                               help='tokens file')
    
    return parser.parse_args()

#  
if __name__ == '__main__':
    
    args = parse_args()
    run_mode = args.mode
    #
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #
    model_tag = args.model_tag
    #
    if model_tag.startswith('transformer'):
        from model_graph_transformer import ModelGraph
    
    #
    # settings & vocab
    settings = ModelSettings()
    settings.model_tag = model_tag
    settings.model_graph = ModelGraph
    settings.gpu_available = args.gpu
    #
    vocab = Vocab()    
    vocab.add_tokens_from_file(args.tokens_file)
    vocab.load_pretrained_embeddings(args.emb_file)
    vocab.emb_dim = settings.emb_dim
    settings.vocab = vocab
    #
    if run_mode == 'predict':
        settings.is_train = False
    else:
        settings.is_train = True
    #
    settings.base_dir = args.base_dir
    settings.check_settings()
    settings.create_or_reset_log_file()
    settings.logger.info('running with args : {}'.format(args))
    settings.logger.info(settings.trans_info_to_dict())
    
    #
    # run
    if run_mode == 'debug':
        model_utils.do_debug(settings, args)
    elif run_mode == 'train':
        model_utils.do_train_and_valid(settings, args)
    elif run_mode == 'eval':
        model_utils.do_eval(settings, args)
    elif run_mode == 'predict':
        model_utils.do_predict(settings, args)
    else:
        print('NOT supported mode. supported modes: debug, train, eval, and predict.')
    #
    
    