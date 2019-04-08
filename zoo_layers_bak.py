# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 17:14:19 2018

@author: limingfan
"""

import tensorflow as tf
from tensorflow.python.ops import array_ops

#
def dropout(inputs, keep_prob, feature_stick=True, mode="recurrent"):
    #
    if feature_stick is False: return tf.nn.dropout(inputs, keep_prob)
    #
    shape = tf.shape(inputs)
    if mode == "embedding" and len(inputs.get_shape().as_list()) == 2:
        noise_shape = [shape[0], 1]
        scale = keep_prob
        out = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape) * scale
    elif mode == "recurrent" and len(inputs.get_shape().as_list()) == 3:     
        noise_shape = [shape[0], 1, shape[-1]]  # batch_major
        out = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)
    else: 
        out = tf.nn.dropout(inputs, keep_prob, noise_shape=None)
    return out

def dense(inputs, hidden, use_bias=True, scope="dense"):
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        shape_list = inputs.get_shape().as_list()
        dim = shape_list[-1]
        out_shape = [shape[idx] for idx in range(len(shape_list) - 1)] + [hidden]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("kernel", [dim, hidden],
                            initializer = tf.variance_scaling_initializer())
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable("bias", [hidden],
                                initializer = tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res
    
def dense_with_w(inputs, hidden, weights, transpose_b=False):
    shape = tf.shape(inputs)
    shape_list = inputs.get_shape().as_list()    
    out_shape = [shape[idx] for idx in range(len(shape_list) - 1)] + [hidden]
    dim = shape_list[-1]
    flat_inputs = tf.reshape(inputs, [-1, dim])
    res = tf.matmul(flat_inputs, weights, transpose_b = transpose_b)
    res = tf.reshape(res, out_shape)
    return res

#
def create_dense_vars(input_size, output_size,
                      use_bias=True, bias_init_value=0.0, scope="dense"):
    with tf.variable_scope(scope):
        W = tf.get_variable("kernel", [input_size, output_size],
                            initializer = tf.variance_scaling_initializer())
        b = None
        if use_bias:
            b = tf.get_variable("bias", [output_size],
                                initializer = tf.constant_initializer(bias_init_value))
        #
        return W, b

def dense_with_vars(inputs, Wb):
    shape = tf.shape(inputs)
    shape_list = inputs.get_shape().as_list()
    if len(shape_list) == 2:
        out = tf.matmul(inputs, Wb[0])
        if Wb[1] is not None:
            out = tf.nn.bias_add(out, Wb[1])
        return out
    else:
        pass
    #
    input_size = shape_list[-1]
    output_size = Wb[0].get_shape().as_list()[1]
    out_shape = [shape[idx] for idx in range(len(shape_list) - 1)] + [output_size]
    #
    flat_inputs = tf.reshape(inputs, [-1, input_size])
    out = tf.matmul(flat_inputs, Wb[0])
    if Wb[1] is not None:
        out = tf.nn.bias_add(out, Wb[1])
    out = tf.reshape(out, out_shape)
    return out

#
def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh((0.79788456 * (x + 0.044715 * tf.pow(x, 3)) )))
    return x * cdf

def layer_norm(x, name=None):
    out = tf.contrib.layers.layer_norm(inputs = x,
                                       begin_norm_axis = -1,
                                       begin_params_axis = -1,
                                       scope = name)
    return out
    
#
def get_posi_emb(input_seq, d_posi_emb, d_model, scope="posi_emb"):
    
    with tf.variable_scope(scope):
        posi = tf.ones_like(input_seq, dtype = tf.float32)
        posi = tf.cumsum(posi, axis = 1) - tf.constant(1.0)
        posi = tf.tile(tf.expand_dims(posi, 2), [1, 1, d_posi_emb])    
        #               
        dim = tf.ones_like(input_seq, dtype = tf.float32)
        dim = tf.tile(tf.expand_dims(dim, 2), [1, 1, d_posi_emb])
        dim = tf.cumsum(dim, axis = 2) - tf.constant(1.0)
        #
        d_model_recip = 1.0/ d_model
        pe = posi * 1e-4**(2 * d_model_recip * dim)
        #
        pe_sin = tf.sin(pe)
        pe_cos = tf.cos(pe)
        #
        pe_sin = tf.concat([pe_sin, pe_cos], -1)
        
    return pe_sin
    
#
def calculate_position_emb_mat(max_seq_len, emb_posi_dim, emb_posi_model,
                               scope = "embedding_mat_posi_scope"):    
    with tf.variable_scope(scope):
        #
        posi = tf.cast(tf.range(max_seq_len), dtype = tf.float32)  # (T,)
        posi = tf.tile(tf.expand_dims(posi, 1), [1, emb_posi_dim])
        #
        dim = tf.cast(tf.range(emb_posi_dim), dtype = tf.float32)  # (D,)
        dim = tf.tile(tf.expand_dims(dim, 0), [max_seq_len, 1])
        #
        d_model_recip = 1.0/ emb_posi_model
        pe = posi * 1e-4**(2 * d_model_recip * dim)   # (T, D)
        #
        pe_sin = tf.sin(pe)
        pe_cos = tf.cos(pe)
        #
        # pe_all = tf.concat([pe_sin, pe_cos], -1)
        #
        pe_sin = tf.expand_dims(pe_sin, -1)  
        pe_cos = tf.expand_dims(pe_cos, -1)
        pe_all = tf.concat([pe_sin, pe_cos], -1)  # (T, D, 2)
        #
        pe_all = tf.reshape(pe_all, [max_seq_len, -1])
        pe_all = pe_all[:, 0:emb_posi_dim]
        
    return pe_all  # pe_sin

#
def qkv_att_layer(query, key, value, mask_mat=None, keep_prob=1.0):
    """ batch_major
        query: [B, TQ, DQ]
        key: [B, TK, DK]    # DQ = DK
        value: [B, TV, DV]  # TK = TV
        mask_mat: [TQ, TK]
        
        return: [B, TQ, DV]
    """
    dim = query.get_shape().as_list()[-1]
    att_mat = tf.matmul(query, tf.transpose(key, [0, 2, 1])) / (dim ** 0.5)
    #
    if mask_mat:
        att_mat = tf.add(att_mat, 1e30 * (mask_mat - 1) )  # -inf   # [B, TQ, TM]
    #
    logits = tf.nn.softmax(att_mat)
    logits = tf.nn.dropout(logits, keep_prob)
    outputs = tf.matmul(logits, value)   # [B, TQ, DV]
    return outputs, logits

def block_transformer(num_layers, num_heads, num_units,
                      inputs, mask=None, keep_prob=1.0, scope="transformer"):
    """
    """
    num_all = num_heads * num_units
    #
    for layer_id in range(num_layers):
        with tf.variable_scope("layer_%d" % layer_id):
            #
            # sublayer-0
            # norm
            inputs = tf.contrib.layers.layer_norm(inputs)
            #
            # multihead attention
            results = []
            for idx in range(num_heads):
                
                qd = tf.nn.dense(inputs, num_units)
                kd = tf.nn.dense(inputs, num_units)
                vd = tf.nn.dense(inputs, num_units)
                
                result, att = qkv_att_layer(qd, kd, vd, mask, keep_prob)
                results.append(result)
            #
            results = tf.concat(results, -1)
            results = tf.nn.dense(results, num_all)
            #
            # drop & add
            results = results + tf.nn.dropout(results, keep_prob)
            #
            # sublayer-1
            # norm
            results = tf.contrib.layers.layer_norm(results)
            #
            # feed forward
            results = tf.layers.dense(results, num_all, activation=tf.nn.relu)
            results = tf.layers.dense(results, num_all)
            #
            # drop & add
            inputs = results + tf.nn.dropout(results, keep_prob)
            #
        #
    #
    return inputs
    #
            

class MultiHeadAttention():
    """
    """
    def __init__(self, num_heads, num_units, keep_prob=1.0, scope="mha"):
        """
        """
        self.keep_prob = keep_prob
        self.num_heads = num_heads
        self.num_units = num_units
        
        d_model = num_heads * num_units
                
        self.query_wb_list = []
        self.key_wb_list = []
        self.value_wb_list = []
        
        with tf.variable_scope(scope):                       
            for idx in range(num_heads):
                query_wb = create_dense_vars(d_model, num_units, "query_wb_%d" % idx)
                key_wb = create_dense_vars(d_model, num_units, "key_wb_%d" % idx)
                value_wb = create_dense_vars(d_model, num_units, "value_wb_%d" % idx)
                self.query_wb_list.append(query_wb)
                self.key_wb_list.append(key_wb)
                self.value_wb_list.append(value_wb)
            #
            self.trans_wb = create_dense_vars(d_model, d_model, "trans_wb") 
            #

    def forward(self, query, key, value, mask=None):
        """
        """
        self.results = []
        self.att_logits = []
        
        for idx in range(self.num_heads):
            
            qd = dense_with_vars(query, self.query_wb_list[idx])
            kd = dense_with_vars(key, self.key_wb_list[idx])
            vd = dense_with_vars(value, self.value_wb_list[idx])
            
            result, att = qkv_att_layer(qd, kd, vd, mask, self.keep_prob)
            
            self.results.append(result)
            self.att_logits.append(att)
            
        results_c = tf.concat(self.results, -1)
        return dense_with_vars(results_c, self.trans_wb)
            
        

class MultiHeadAttention_Bak():
    """
    """
    def __init__(self, num_heads, num_units, keep_prob=1.0, scope="mha"):
        """
        """
        self.keep_prob = keep_prob
        self.num_heads = num_heads
        self.num_units = num_units
        
        d_model = num_heads * num_units
        
        with tf.variable_scope(scope):
            self.query_wb = create_dense_vars(d_model, d_model, "query_wb")
            self.key_wb = create_dense_vars(d_model, d_model, "key_wb")
            self.value_wb = create_dense_vars(d_model, d_model, "value_wb")
            self.trans_wb = create_dense_vars(d_model, d_model, "trans_wb")
        
    def forward(self, query, key, value, mask=None):
        """
        """
        query_d = dense_with_vars(query, self.query_wb)
        key_d = dense_with_vars(key, self.key_wb)
        value_d = dense_with_vars(value, self.value_wb)
        
        query_s = array_ops.split(value = query_d,
                                  num_or_size_splits = self.num_heads, axis = -1)
        key_s = array_ops.split(value = key_d,
                                num_or_size_splits = self.num_heads, axis = -1)
        value_s = array_ops.split(value = value_d,
                                  num_or_size_splits = self.num_heads, axis = -1)
        
        query_e = tf.concat([tf.expand_dims(item, 0) for item in query_s], 0)
        key_e = tf.concat([tf.expand_dims(item, 0) for item in key_s], 0)
        value_e = tf.concat([tf.expand_dims(item, 0) for item in value_s], 0)
        
        return query_e, key_e, value_e

    
#
def att_qkv_layer_bak(inputs, memory, values, mask_m, att_dim, keep_prob=1.0, scope="qkv"):
    """ batch_major
        inputs: [B, TQ, DQ]
        memory: [B, TM, DM]
        values: [B, TV, DV]  # TM = TV
    """
    with tf.variable_scope(scope):
        d_inputs = dropout(inputs, keep_prob=keep_prob)  # [B, TQ, DQ]
        d_memory = dropout(memory, keep_prob=keep_prob)
        #
        inputs_d = dense(d_inputs, att_dim, use_bias=False, scope="inputs")            
        memory_d = dense(d_memory, att_dim, use_bias=False, scope="memory")
        # inputs_d = tf.nn.relu(inputs_d)
        # memory_d = tf.nn.relu(memory_d)
        #
        # [B, TQ, TM]
        att_mat = tf.matmul(inputs_d, tf.transpose(memory_d, [0, 2, 1])) / (att_dim ** 0.5)
        # 
        mask_3d = tf.cast(tf.expand_dims(mask_m, axis=1), tf.float32) # [B, 1, TM]
        att_masked = tf.add(att_mat, 1e30 * (mask_3d - 1) )  # -inf   # [B, TQ, TM]
        logits = tf.nn.softmax(att_masked)
        #
        d_values = dropout(values, keep_prob=keep_prob)  # [B, TM, DV]
        values_d = dense(d_values, att_dim, use_bias=False, scope="values")
        # values_d = tf.nn.relu(values_d)
        #
        outputs = tf.matmul(logits, values_d)   # [B, TQ, DV_d]
    return outputs
    
def qk_mat_layer(inputs, memory, att_dim, keep_prob=1.0, scope="qk_mat"):
    with tf.variable_scope(scope):
        d_inputs = dropout(inputs, keep_prob=keep_prob)  # [B, TQ, D]
        d_memory = dropout(memory, keep_prob=keep_prob)
        #
        inputs_d = dense(d_inputs, att_dim, use_bias=False, scope="inputs")            
        memory_d = dense(d_memory, att_dim, use_bias=False, scope="memory")
        # inputs_d = tf.nn.relu(inputs_d)
        # memory_d = tf.nn.relu(memory_d)
        #
        # [B, TQ, TM]
        att_mat = tf.matmul(inputs_d, tf.transpose(memory_d, [0, 2, 1])) / (att_dim ** 0.5)
    return att_mat

def qk_value_pool_layer(qk_mat, values, mask_k, hidden, keep_prob=1.0, scope="qk_pool"):
    with tf.variable_scope(scope):
        # 
        mask_3d = tf.cast(tf.expand_dims(mask_k, axis=1), tf.float32) # [B, 1, TM]
        att_masked = tf.add(qk_mat, 1e30 * (mask_3d - 1) )  # -inf   # [B, TQ, TM]
        logits = tf.nn.softmax(att_masked)
        #
        d_values = dropout(values, keep_prob=keep_prob)  # [B, TM, DV]
        values_d = dense(d_values, qk_mat, use_bias=False, scope="values")
        # values_d = tf.nn.relu(values_d)
        outputs = tf.matmul(logits, values_d)   # [B, TQ, DV_d]
    return outputs

#
def do_mask_padding_elems(x, mask):
    # make padding elements in x to -inf,
    # for next step of softmax,
    # (batch, time), or (time, batch), or
    # (batch, time, units), or (time, batch, units)
    return tf.add(x, 1e30 * tf.cast(mask - 1, dtype=tf.float32) )

def att_pool_layer(query, seq, seq_mask, att_dim, keep_prob=1.0, scope="att_pooling"):
    """ batch_major
        query: [B, DQ]
        seq: [B, TM, DM]
        seq_mask: [B, TM]
    """
    with tf.variable_scope(scope):
        #
        query = tf.expand_dims(query, 1)  # [B, 1, DQ], TQ = 1
        #
        d_inputs = dropout(query, keep_prob=keep_prob)  # [B, TQ, DQ]
        d_memory = dropout(seq, keep_prob=keep_prob)    # [B, TM, DM]
        #
        inputs_d = dense(d_inputs, att_dim, use_bias=False, scope="inputs")            
        memory_d = dense(d_memory, att_dim, use_bias=False, scope="memory")
        # inputs_d = tf.nn.relu(inputs_d)
        # memory_d = tf.nn.relu(memory_d)
        #
        # [B, TQ, TM]        
        att_mat = tf.matmul(inputs_d, tf.transpose(memory_d, [0, 2, 1])) / (att_dim ** 0.5)
        # 
        mask_3d = tf.cast(tf.expand_dims(seq_mask, axis=1), tf.float32) # [B, 1, TM]
        att_masked = tf.add(att_mat, 1e30 * (mask_3d - 1) )  # -inf   # [B, TQ, TM]
        logits = tf.nn.softmax(att_masked)
        #
        d_values = dropout(seq, keep_prob=keep_prob)  # [B, TM, DV]
        values_d = dense(d_values, att_dim, use_bias=False, scope="values")
        # values_d = tf.nn.relu(values_d)
        #
        outputs = tf.matmul(logits, values_d)   # [B, TQ, DV_d]
        outputs = tf.squeeze(outputs, 1)        # [B, DV_d]
    return outputs

#
def rnn_layer(input_sequence, sequence_length, rnn_size,
              keep_prob = 1.0, activation = None,
              concat = True, scope = 'bi-lstm'):
    '''build bidirectional lstm layer'''
    #
    # time_major = False
    #
    # input_sequence = tf.nn.dropout(input_sequence, keep_prob)
    input_sequence = dropout(input_sequence, keep_prob)
    #
    weight_initializer = tf.truncated_normal_initializer(stddev = 0.01)
    act = activation or tf.nn.tanh
    #
    cell_fw = tf.contrib.rnn.LSTMCell(rnn_size, activation = act,
                                      initializer = weight_initializer)
    cell_bw = tf.contrib.rnn.LSTMCell(rnn_size, activation = act,
                                      initializer = weight_initializer)
    #
    #cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=dropout_rate)
    #cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=dropout_rate)
    #
    #cell_fw = MyLSTMCell(rnn_size, keep_prob, initializer = weight_initializer)
    #cell_bw = MyLSTMCell(rnn_size, keep_prob, initializer = weight_initializer)
    #
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_sequence,
                                                    sequence_length = sequence_length,
                                                    time_major = False,
                                                    dtype = tf.float32,
                                                    scope = scope)
    #
    if concat:
        rnn_output = tf.concat(rnn_output, 2, name = 'output')
    else:
        rnn_output = tf.multiply(tf.add(rnn_output[0], rnn_output[1]), 0.5, name = 'output')
    #
    return rnn_output
    #

def gru_layer(input_sequence, sequence_length, rnn_size,
              keep_prob = 1.0, activation = None,
              concat = True, scope = 'bi-gru'):
    '''build bidirectional gru layer'''
    #
    # time_major = False
    #
    # input_sequence = tf.nn.dropout(input_sequence, keep_prob)
    input_sequence = dropout(input_sequence, keep_prob)
    #
    act = activation or tf.nn.tanh
    #
    cell_fw = tf.nn.rnn_cell.GRUCell(rnn_size, activation = act)
    cell_bw = tf.nn.rnn_cell.GRUCell(rnn_size, activation = act)
    #
    # cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=dropout_rate)
    # cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=dropout_rate)
    #
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_sequence,
                                                    sequence_length = sequence_length,
                                                    time_major = False,
                                                    dtype = tf.float32,
                                                    scope = scope)
    #
    if concat:
        rnn_output = tf.concat(rnn_output, 2, name = 'output')
    else:
        rnn_output = tf.multiply(tf.add(rnn_output[0], rnn_output[1]), 0.5, name = 'output')
    #
    return rnn_output
    #     

#
def gather_and_pad_layer(x, num_items):
    """ x: (BS', D)
        num_items : (B,)
        
        returning: (B, S, D), (B, S)
    """
    B = tf.shape(num_items)[0]
    T = tf.reduce_max(num_items)
    
    pad_item = tf.zeros(shape = tf.shape(x[0:1,:]) )
    one_int32 = tf.ones(shape = (1,), dtype = tf.int32)
    zero_int32 = tf.zeros(shape = (1,), dtype = tf.int32)
    
    bsd_ta = tf.TensorArray(size = B, dtype = tf.float32)
    mask_ta = tf.TensorArray(size = B, dtype = tf.int32)
    time = tf.constant(0)
    posi = tf.constant(0)
    
    def condition(time, posi_s, bsd_s, mask_s):
        return tf.less(time, B)
    
    def body(time, posi_s, bsd_s, mask_s):        
        posi_e = posi_s + num_items[time]        
        chunk = x[posi_s:posi_e, :]
        #
        mask_c = tf.tile(one_int32, [ num_items[time] ] )
        #
        d = T - num_items[time]
        chunk, mask_c = tf.cond(d > 0,
                                lambda: (tf.concat([chunk, tf.tile(pad_item, [d, 1])], 0),
                                         tf.concat([mask_c, tf.tile(zero_int32, [d])], 0) ),
                                lambda: (chunk, mask_c) )
        #
        bsd_s = bsd_s.write(time, chunk)
        mask_s = mask_s.write(time, mask_c)
        return (time + 1, posi_e, bsd_s, mask_s)
        
    t, p, bsd_w, mask_w = tf.while_loop(cond = condition, body = body,
                                        loop_vars = (time, posi, bsd_ta, mask_ta) )
    bsd = bsd_w.stack()
    mask = mask_w.stack()
    
    return bsd, mask

