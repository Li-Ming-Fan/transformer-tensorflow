# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:01:52 2019

@author: limingfan
"""

import tensorflow as tf

from zoo_layers import get_mask_mat_subsequent


#
def do_greedy_decoding(model, src, src_mask, max_len, start_symbol_id):
    """
    """
    memory = model.encode(src, src_mask)
    mask_subsequent = get_mask_mat_subsequent(max_len)
    
    batch_start = tf.cast(tf.ones_like(src[:, 0:1]), dtype = tf.int32)
    batch_start = tf.multiply(batch_start, start_symbol_id)    
    bp = tf.cast(tf.zeros_like(src[:, 0:1]), dtype = tf.int32)
    
    logits_list = []
    preds_list = []
    
    rem_len = max_len - 1
    dc_feed = tf.concat([batch_start, tf.tile(bp, [0, rem_len])], 1)
    for step in range(max_len):
        out = model.decode(memory, src_mask, dc_feed, mask_subsequent)
        logits_curr = model.generator.forward(out)        
        logits_curr = tf.nn.softmax(logits_curr, -1)     
        preds_curr = tf.nn.argmax(logits_curr, -1)
        
        logits_list.append(logits_curr)
        preds_list.append(preds_curr)
        #
        preds = tf.transpose(tf.stack(preds_list, 0), [1, 0])
        #
        rem_len -= 1
        if rem_len > 0:
            dc_feed = tf.concat([batch_start, preds, tf.tile(bp, [0, rem_len])], 1)
        elif rem_len == 0: # step = max_len - 2
            dc_feed = tf.concat([batch_start, preds], 1)
        else:  # step = max_len - 1
            pass
        #
    #
    logits = tf.transpose(tf.stack(logits_list, 0), [1, 0, 2])    
    return logits, preds


def do_beam_search_decoding(model, src, src_mask, max_len, start_symbol_id, beam_width):
    """
    """
    
    pass

