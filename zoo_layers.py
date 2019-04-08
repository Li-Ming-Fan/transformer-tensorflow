# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 17:14:19 2018

@author: limingfan
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops


class Dense():
    """
    """
    def __init__(self, input_size, output_size,
                 use_bias=True, bias_init_value=0, scope="dense"):
        """
        """        
        with tf.variable_scope(scope):
            W = tf.get_variable("kernel", [input_size, output_size],
                                initializer = tf.variance_scaling_initializer())
            b = None
            if use_bias:
                b = tf.get_variable("bias", [output_size],
                                    initializer = tf.constant_initializer(bias_init_value))
            #
            self.w = W
            self.b = b
            #
    
    def __call__(self, inputs):
        """
        """        
        shape = tf.shape(inputs)
        shape_list = inputs.get_shape().as_list()
        if len(shape_list) == 2:
            out = tf.matmul(inputs, self.w)
            if self.b is not None:
                out = tf.nn.bias_add(out, self.b)
            return out
        else:
            pass
        #
        input_size = shape_list[-1]
        output_size = self.w.get_shape().as_list()[1]
        out_shape = [shape[idx] for idx in range(len(shape_list) - 1)] + [output_size]
        #
        flat_inputs = tf.reshape(inputs, [-1, input_size])
        out = tf.matmul(flat_inputs, self.w)
        if self.b is not None:
            out = tf.nn.bias_add(out, self.b)
        out = tf.reshape(out, out_shape)
        return out

class LayerNorm():
    """
    """
    def __init__(self, num_units, epsilon=1e-6, scope="layer_norm"):
        """
        """
        with tf.variable_scope(scope):
            self.beta = tf.get_variable('beta', [num_units],
                                        initializer=tf.ones_initializer(),
                                        trainable=True)
            self.gamma = tf.get_variable('gamma', [num_units],
                                         initializer=tf.zeros_initializer(),
                                         trainable=True)
            self.eps = epsilon
    
    def __call__(self, x):
        """
        """
        mean, std = tf.nn.moments(x, [-1], keep_dims=True, name='moments')        
        return self.beta * (x - mean)/ (std + self.eps) + self.gamma
    
class Dropout():
    """
    """
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
        
    def __call__(self, x):        
        return tf.nn.dropout(x, self.keep_prob)
    
# 
def build_module_copies(module_class, class_args, N, scope="module_copies"):
    """ module_class: must be some class
        class_args: be the args for module_class init function
    """
    list_copies = []
    with tf.variable_scope(scope):
        for idx in range(N):
            with tf.variable_scope("copy_%d" % idx):
                module_copied = module_class(*class_args)
                list_copies.append(module_copied)
    return list_copies

#
def qkv_att_layer(query, key, value, mask_mat=None, keep_prob=1.0):
    """ batch_major
        query: [B, TQ, DQ]
        key: [B, TK, DK]    # DQ = DK
        value: [B, TV, DV]  # TK = TV
        mask_mat: [B, TQ, TK], or [1, TQ, TK]
        
        return: [B, TQ, DV]
    """
    dim = query.get_shape().as_list()[-1]
    att_mat = tf.matmul(query, key, transpose_b=True) / (dim ** 0.5)
    #
    if mask_mat:
        att_mat = tf.add(att_mat, 1e30 * (mask_mat - 1) )  # -inf   # [B, TQ, TM]
    #
    logits = tf.nn.softmax(att_mat)
    logits = tf.nn.dropout(logits, keep_prob)
    outputs = tf.matmul(logits, value)   # [B, TQ, DV]
    return outputs, logits

#
class MultiHeadAttention():
    """
    """
    def __init__(self, num_heads, num_units, keep_prob=1.0, scope="multi_head"):
        """
        """
        self.keep_prob = keep_prob
        self.num_heads = num_heads
        self.num_units = num_units
        
        self.attention = None
        
        d_model = num_heads * num_units
        with tf.variable_scope(scope):
            self.dense_query = Dense(d_model, d_model, "dense_query")
            self.dense_key = Dense(d_model, d_model, "dense_key")
            self.dense_value = Dense(d_model, d_model, "dense_value")
            self.dense_trans = Dense(d_model, d_model, "dense_trans")
            
        
    def __call__(self, query, key, value, mask_mat=None):
        """
        """
        query_d = self.dense_query(query)
        key_d = self.dense_key(key)
        value_d = self.dense_value(value)
        
        query_s = array_ops.split(value = query_d,
                                  num_or_size_splits = self.num_heads, axis = -1)
        key_s = array_ops.split(value = key_d,
                                num_or_size_splits = self.num_heads, axis = -1)
        value_s = array_ops.split(value = value_d,
                                  num_or_size_splits = self.num_heads, axis = -1)
        
        # [B, H, T, D]
        query_e = tf.concat([tf.expand_dims(item, 1) for item in query_s], 1)
        key_e = tf.concat([tf.expand_dims(item, 1) for item in key_s], 1)
        value_e = tf.concat([tf.expand_dims(item, 1) for item in value_s], 1)
        
        # qkv
        if mask_mat is None:
            mask_mat_e = None
        else:
            mask_mat_e = tf.expand_dims(mask_mat, 1)
        #
        out, att = qkv_att_layer(query_e, key_e, value_e, mask_mat_e, keep_prob=1.0)
        #
        # concat & linear
        out_list = [ out[:,idx,:,:] for idx in range(self.num_heads) ]
        out_c = tf.concat(out_list, -1)
        out_d = self.dense_trans(out_c)
        #
        self.attention = att
        return out_d
    
class PositionwiseFeedForward():
    """
    """
    def __init__(self, num_dim_all, dim_middle, keep_prob, scope="pwff"):
        """
        """        
        with tf.variable_scope(scope):
            self.d1 = Dense(num_dim_all, dim_middle, scope="ff_d1")
            self.d2 = Dense(dim_middle, num_dim_all, scope="ff_d2")
            self.dropout = Dropout(keep_prob)
        
    def __call__(self, x):
        x = tf.nn.relu(self.d1(x))
        x = self.d2(self.dropout(x))
        return x

#
def SublayerWrapper():
    """
    """
    def __init__(self, num_units, keep_prob, sublayer_class, class_args,
                 scope="sublayer_wrapper"):
        """
        """
        with tf.variable_scope(scope):
            self.layer_norm = LayerNorm(num_units)
            self.dropout = Dropout(keep_prob)
            self.sublayer = sublayer_class(*class_args)
    
    def __call__(self, x, sublayer_invoker):
        """
        """
        return x + self.dropout(sublayer_invoker(self.layer_norm(x)))
    
#
def get_mask_mat_from_mask_seq(mask):
    """ mask: [B, T]
    """
    mask = tf.cast(tf.expand_dims(mask, 1), tf.float32)
    mask_rows = tf.transpose(mask, [0, 2, 1])
    mask = mask * mask_rows
    return mask

#
def get_mask_mat_subsequent(size):
    """
    """    
    mask_mat = np.zeros((1, size, size), dtype = np.float32)
    for idx in range(size):
        for idy in range(size):
            if idx <= idy: mask_mat[0, idx, idy] = 1.0
    #
    mask_tensor = tf.get_variable("mask_subsequent",
                                  shape = (1, size, size),
                                  initializer = tf.constant_initializer(mask_mat),
                                  trainable = False)
    return mask_tensor
    
#
def calculate_position_emb_mat(max_seq_len, posi_emb_dim, posi_emb_model):
    """
    """
    d_model_recip_2 = 2.0 / posi_emb_model
    
    arg_mat = np.zeros((max_seq_len, posi_emb_dim), dtype=np.float32)
    for idx in range(max_seq_len):
        for idm in range(posi_emb_dim):
            arg_mat[idx, idm] = idx * 1e-4**(d_model_recip_2 * idm)
    #
    pe_sin = np.sin(arg_mat)
    pe_cos = np.cos(arg_mat)
    #    
    pe_sin = np.expand_dims(pe_sin, -1)  
    pe_cos = np.expand_dims(pe_cos, -1)
    pe_all = np.concat([pe_sin, pe_cos], -1)  # (T, D, 2)
    #
    pe_all = np.reshape(pe_all, [max_seq_len, -1])
    pe_all = pe_all[:, 0:posi_emb_dim]
    #
    # tf.Tensor
    pe_mat = tf.get_variable("position_embeddings",
                             shape = (max_seq_len, posi_emb_dim),
                             initializer = tf.constant_initializer(pe_all),
                             trainable = False)
        
    return pe_mat

def get_emb_positioned(x, token_emb, position_emb):
    """ x: [None, None]
    """
    posi = tf.range(tf.shape(x)[-1])
    
    seq_emb_t = tf.nn.embedding_lookup(token_emb, x)
    seq_emb_p = tf.nn.embedding_lookup(position_emb, posi)
    
    return seq_emb_t + seq_emb_p

#
def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh((0.79788456 * (x + 0.044715 * tf.pow(x, 3)) )))
    return x * cdf

