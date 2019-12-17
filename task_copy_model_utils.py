# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 05:08:20 2019

@author: limingfan
"""

import os
import numpy as np

from Zeras.data_batcher import DataBatcher
from Zeras.model_wrapper import ModelWrapper

import task_copy_data_set as data_set

#
def eval_process(model, eval_batcher, max_batches_eval, mode_eval):
    """
    """
    loss_aver, metric_aver = 0.0, 0.0
    count = 0
    while True:
        batch = eval_batcher.get_next_batch()  
        #
        if batch is None: break
        if count == max_batches_eval: continue  #
        #
        count += 1
        # print(count)
        #
        result_dict = model.run_eval_one_batch(batch)
        loss = result_dict["loss_optim"]
        metric = result_dict["metric"]
        #
        loss_aver += loss
        metric_aver += metric
        # print(loss)
        # print(metric)
        #
        if mode_eval:
            print(count)
            print("batch data:")
            print(batch["labels_seq"])
            #
            print("results:")
            print(np.argmax(result_dict["logits"], -1) )
            print()        
        #
    #
    loss_aver /= count
    # metric_aver /= count
    metric_aver = 100 - loss_aver
    #
    model.logger.info('eval finished, with total num_batches: %d' % count)
    # model.logger.info('loss_aver, metric_aver: %g, %g' % (loss_aver, metric_aver))
    #
    eval_score = {}
    return eval_score, loss_aver, metric_aver
    #
    
def do_eval(settings, args):
    #
    if args.ckpt_loading == "latest":
        dir_ckpt = settings.model_dir
    else:
        dir_ckpt = settings.model_dir_best
    #
    # model
    model = settings.ModelClass(settings)
    model.prepare_for_train_and_valid(dir_ckpt)
    model.assign_dropout_keep_prob(1.0)
    #
    # data
    data_batcher = DataBatcher(data_set.get_examples_generator(),
                               data_set.batch_std_transor,
                               settings.batch_size_eval, single_pass = True,
                               worker_type="thread")
    #
    # eval
    eval_score, loss_aver, metric_aver = eval_process(model, data_batcher,
                                                      settings.max_batches_eval,
                                                      mode_eval = True)
    #
    print('loss_aver, metric_aver: %g, %g' % (loss_aver, metric_aver))
    model.logger.info('loss_aver, metric_aver: %g, %g' % (loss_aver, metric_aver))
    model.logger.info('{}'.format(eval_score))
    #
    
def do_train_and_valid(settings, args):
    #
    if args.ckpt_loading == "latest":
        dir_ckpt = settings.model_dir
    else:
        dir_ckpt = settings.model_dir_best
    #
    # model
    model = settings.ModelClass(settings)
    model.prepare_for_train_and_valid(dir_ckpt)
    #    
    # data
    data_batcher = DataBatcher(data_set.get_examples_generator(),
                               data_set.batch_std_transor,
                               settings.batch_size, single_pass = False,
                               worker_type="thread")
    eval_period = settings.valid_period_batch
    #
    # train
    loss = 10000.0
    best_metric_val = 0
    # last_improved = 0
    lr = 0.0
    #
    count = 0
    model.logger.info("")
    while True:
        #
        # eval
        if count % eval_period == 0:            
            model.logger.info("training curr batch, loss, lr: %d, %g, %g" % (count, loss, lr) )
            #
            model.save_ckpt(settings.model_dir, settings.model_name, count)
            model.assign_dropout_keep_prob(1.0)
            #
            model.logger.info('evaluating after num_batches: %d' % count)
            eval_batcher = DataBatcher(data_set.get_examples_generator(),
                                       data_set.batch_std_transor,
                                       settings.batch_size, single_pass = True,
                                       worker_type="thread")
            #
            eval_score, loss_aver, metric_val = eval_process(model, eval_batcher,
                                                             settings.max_batches_eval,
                                                             mode_eval = False)
            model.logger.info("eval loss_aver, metric, metric_best: %g, %g, %g" % (
                    loss_aver, metric_val, best_metric_val) )
            #
            # save best
            if metric_val >= best_metric_val:  # >=
                best_metric_val = metric_val
                # last_improved = count
                # ckpt
                model.logger.info('a new best model, saving ...')
                model.save_ckpt_best(settings.model_dir_best, settings.model_name, count)
                #
            """
            # decay
            if count - last_improved >= model.patience_decay:
                lr *= model.ratio_decay
                model.assign_learning_rate(lr)
                last_improved = count
                model.logger.info('learning_rate decayed after num_batches: %d' % count)
                model.logger.info('current learning_rate %g' % lr)
                #
                if lr < model.learning_rate_minimum:
                    model.logger.info('current learning_rate < learning_rate_minimum, stop training')
                    break
                #
            """
            #
            # lr *= model.ratio_decay
            # model.assign_learning_rate(lr)
            # model.logger.info('learning_rate decayed after num_batches: %d' % count)
            # model.logger.info('current learning_rate: %g' % lr)
            #
            if lr < settings.learning_rate_minimum and count > settings.warmup_steps:
                model.logger.info('current learning_rate < learning_rate_minimum, stop training')
                break
            #
            model.assign_dropout_keep_prob(settings.keep_prob)
            model.logger.info("")
            #
        #
        # train
        batch = data_batcher.get_next_batch()  
        # if batch is None: break
        count += 1
        # print(count)        
        #
        result_dict = model.run_train_one_batch(batch)   # just for train
        loss = result_dict["loss_optim"]
        lr = result_dict["lr"]
        #
        # print(loss)
        # model.logger.info("training curr batch, loss, lr: %d, %g, %g" % (count, loss, lr)
        #
    #
    model.logger.info("training finshed with total num_batches: %d" % count)
    #
    
def do_predict(settings, args):
    #
    if args.ckpt_loading == "latest":
        dir_ckpt = settings.model_dir
    else:
        dir_ckpt = settings.model_dir_best
    #
    pb_file = os.path.join(dir_ckpt, "model_frozen.pb")
    #
    # model
    model = settings.ModelClass(settings)
    model.prepare_for_prediction_with_pb(pb_file)
    #
    # data
    data_batcher = DataBatcher(data_set.get_examples_generator(),
                               data_set.batch_std_transor,
                               settings.batch_size_eval, single_pass = True,
                               worker_type="thread")
    #
    # predict
    count = 0
    while True:
        batch = data_batcher.get_next_batch()  
        #
        if batch is None: break
        if count == settings.max_batches_eval: continue  #
        #
        count += 1
        print(count)
        #
        print("batch data:")
        print(batch["labels_seq"])
        print("batch data end")
        #
        result_dict = model.predict_with_pb_from_batch(batch)
        #
        print("results:")
        print(np.argmax(result_dict["logits"], -1) )
        print("results end")
        print()
        #
    #
    model.logger.info('prediction finished, with total num_batches: %d' % count)
    #

def do_convert(settings, args):
    #
    if args.ckpt_loading == "latest":
        dir_ckpt = settings.model_dir
    else:
        dir_ckpt = settings.model_dir_best
    #
    # pb_file = os.path.join(dir_ckpt, "model_saved.pb")
    #
    # model
    model = settings.ModelClass(settings)
    settings.ModelClass.load_ckpt_and_save_pb_file(model, dir_ckpt)
    #