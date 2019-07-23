# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 22:12:00 2019

@author: li-ming-fan
"""

import tensorflow as tf

from zoo_layers import dense, layer_norm
from zoo_layers import multihead_attention_layer

from zoo_nn import gelu


"""
model components of transformer

"""
    
def do_projection(x, vocab_size, emb_mat=None, scope="proj"):
    """
    """
    with tf.variable_scope(scope):
        if emb_mat is None:
            out = dense(x, vocab_size)
        else:
            out = dense(x, vocab_size, emb_mat, transpose_b=True)
        return out
    
def do_encoding(settings, x, mask_mat, keep_prob):
    """
    """
    seq_input = layer_norm(x, scope="encoder_layer_norm_start")
    
    num_layers = settings.num_layers
    num_heads = settings.num_heads
    num_units = settings.num_units
    
    dim_all = num_heads * num_units
    dim_middle = settings.dim_ffm
    activation_type = settings.activation
    
    for layer_id in range(num_layers):        
        with tf.variable_scope("encoder_layer_%d" % layer_id):
        
            # sublayer-0
            # att
            seq_att = multihead_attention_layer(num_heads, num_units,
                                                seq_input, seq_input, seq_input,
                                                mask_mat, keep_prob,
                                                scope = "mh_att")
            #
            # drop
            seq_drop = tf.nn.dropout(seq_att, keep_prob=keep_prob)
            #
            # add & norm
            seq_1 = layer_norm(seq_input + seq_drop, "layer_norm_att")
            #
            
            # sublayer-1
            # dense
            if activation_type == "relu":
                act = tf.nn.relu
            else:
                act = gelu
            #
            seq_d = tf.layers.dense(seq_1, dim_middle, activation = act)
            seq_d = tf.layers.dense(seq_d, dim_all)
            #
            # drop
            seq_drop = tf.nn.dropout(seq_d, keep_prob=keep_prob)
            #
            # add & norm
            seq_input = layer_norm(seq_1 + seq_drop, "layer_norm_ff")
            #
    #
    return seq_input
    #

def do_decoding_one_step(settings, x, mask_dcd, memory, mask_crs, keep_prob):
    """
    """
    seq_input = layer_norm(x, scope="decoder_layer_norm_start")
    
    num_layers = settings.num_layers
    num_heads = settings.num_heads
    num_units = settings.num_units
    
    dim_all = num_heads * num_units
    dim_middle = settings.dim_ffm    
    activation_type = settings.activation
    
    for layer_id in range(num_layers):        
        with tf.variable_scope("decoder_layer_%d" % layer_id):
        
            # sublayer-0
            # att
            seq_att = multihead_attention_layer(num_heads, num_units,
                                                seq_input, seq_input, seq_input,
                                                mask_dcd, keep_prob,
                                                scope = "mh_self_att")
            #
            # drop
            seq_drop = tf.nn.dropout(seq_att, keep_prob=keep_prob)
            #
            # add & norm
            seq_1 = layer_norm(seq_input + seq_drop, "layer_norm_att")
            #
            
            # sublayer-1
            # att
            seq_att = multihead_attention_layer(num_heads, num_units,
                                                seq_1, memory, memory,
                                                mask_crs, keep_prob,
                                                scope = "mh_cross_att")
            #
            # drop
            seq_drop = tf.nn.dropout(seq_att, keep_prob=keep_prob)
            #
            # add & norm
            seq_2 = layer_norm(seq_1 + seq_drop, "layer_norm_cross")
            #
            
            # sublayer-2
            # dense
            if activation_type == "relu":
                act = tf.nn.relu
            else:
                act = gelu
            #
            seq_d = tf.layers.dense(seq_2, dim_middle, activation = act)
            seq_d = tf.layers.dense(seq_d, dim_all)
            #
            # drop
            seq_drop = tf.nn.dropout(seq_d, keep_prob=keep_prob)
            #
            # add & norm
            seq_input = layer_norm(seq_2 + seq_drop, "layer_norm_ff")
            #
    #
    return seq_input
    #
    
#
def do_greedy_decoding(do_decoding_one_step, do_projection, max_len,
                       src_encoded, sub_mask, crs_mask, start_symbol_id):
    """
    """
    memory = src_encoded
    
    batch_start = tf.cast(tf.ones_like(src_encoded[:, 0:1]), dtype = tf.int64)
    batch_start = tf.multiply(batch_start, start_symbol_id)    
    pad_tokens = tf.cast(tf.zeros_like(src_encoded[:, 0:1]), dtype = tf.int64)
    pad_tokens = tf.tile(pad_tokens, [1, max_len-1])
    
    logits_list = []
    preds_list = []
    
    dcd_feed = tf.concat([batch_start, pad_tokens], 1)
    for step in range(max_len):        
        out = do_decoding_one_step(dcd_feed, sub_mask, memory, crs_mask)
        out_last = out[:, step, :]
        logits_curr = do_projection(out_last)
        logits_curr = tf.nn.softmax(logits_curr, -1)
        preds_curr = tf.argmax(logits_curr, -1)
        
        logits_list.append(logits_curr)
        preds_list.append(preds_curr)
        #
        preds = tf.transpose(tf.stack(preds_list, 0), [1, 0])
        dcd_feed = tf.concat([batch_start, preds, pad_tokens[:, step+1:]], 1)
        #
        tf.get_variable_scope().reuse_variables()
        #
    #
    logits = tf.transpose(tf.stack(logits_list, 0), [1, 0, 2])
    return logits, preds


def do_beam_search_decoding(do_decoding_one_step, do_projection, max_len,
                            src_encoded, sub_mask, crs_mask,
                            start_symbol_id, beam_width):
    """
    """
    
    pass

