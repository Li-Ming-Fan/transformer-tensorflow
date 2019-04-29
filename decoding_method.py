# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:01:52 2019

@author: limingfan
"""

import tensorflow as tf

#
def do_greedy_decoding(model, src, src_mask, max_len,
                       sub_mask, crs_mask, start_symbol_id):
    """
    """
    memory = model.encode(src, src_mask)
    
    batch_start = tf.cast(tf.ones_like(src[:, 0:1]), dtype = tf.int64)
    batch_start = tf.multiply(batch_start, start_symbol_id)    
    pad_tokens = tf.cast(tf.zeros_like(src[:, 0:1]), dtype = tf.int64)
    pad_tokens = tf.tile(pad_tokens, [1, max_len-1])
    
    logits_list = []
    preds_list = []
    
    dcd_feed = tf.concat([batch_start, pad_tokens], 1)
    for step in range(max_len):        
        out = model.decode(dcd_feed, sub_mask, memory, crs_mask)
        out_last = out[:, step, :]
        logits_curr = model.generator.forward(out_last)
        logits_curr = tf.nn.softmax(logits_curr, -1)
        preds_curr = tf.argmax(logits_curr, -1)
        
        logits_list.append(logits_curr)
        preds_list.append(preds_curr)
        #
        preds = tf.transpose(tf.stack(preds_list, 0), [1, 0])
        dcd_feed = tf.concat([batch_start, preds, pad_tokens[:, step+1:]], 1)
        #
    #
    logits = tf.transpose(tf.stack(logits_list, 0), [1, 0, 2])
    return logits, preds


def do_beam_search_decoding(model, src, src_mask, max_len,
                            sub_mask, crs_mask, start_symbol_id, beam_width):
    """
    """
    
    pass

