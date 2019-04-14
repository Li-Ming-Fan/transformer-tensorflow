# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:01:52 2019

@author: limingfan
"""

import tensorflow as tf

#
def do_greedy_decoding(model, src, src_mask, max_len,
                       subs_masks, dcd_crs_masks, start_symbol_id):
    """
    """
    memory = model.encode(src, src_mask)
    
    batch_start = tf.cast(tf.ones_like(src[:, 0:1]), dtype = tf.int64)
    batch_start = tf.multiply(batch_start, start_symbol_id)
    
    logits_list = []
    preds_list = []
    
    dcd_feed = batch_start
    for step in range(max_len):
        mask_subsequent = subs_masks[step]
        crs_mask = dcd_crs_masks[step]
        out = model.decode(dcd_feed, mask_subsequent, memory, crs_mask)
        out_last = out[:, -1, :]
        logits_curr = model.generator.forward(out_last)
        logits_curr = tf.nn.softmax(logits_curr, -1)
        preds_curr = tf.argmax(logits_curr, -1)
        
        logits_list.append(logits_curr)
        preds_list.append(preds_curr)
        #
        preds = tf.transpose(tf.stack(preds_list, 0), [1, 0])
        dcd_feed = tf.concat([batch_start, preds], 1)
        #
    #
    logits = tf.transpose(tf.stack(logits_list, 0), [1, 0, 2])    
    return logits, preds


def do_beam_search_decoding(model, src, src_mask, max_len,
                            subs_masks, dcd_crs_masks, start_symbol_id, beam_width):
    """
    """
    
    pass

