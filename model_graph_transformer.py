# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 07:18:29 2019

@author: limingfan
"""

import tensorflow as tf

from zoo_layers import calculate_position_emb_mat, get_emb_positioned
from zoo_layers import get_mask_mat_from_mask_seq

from encoder_decoder import EncoderDecoder, Generator
from encoder_decoder import Encoder, Decoder
from encoder_decoder import EncoderLayer, DecoderLayer


class ModelGraph():
    
    @staticmethod
    def build_placeholder(settings):
        
        inputs_seq = tf.placeholder(tf.int32, [None, None], name='inputs_seq')  # id in vocab
        inputs_mask = tf.placeholder(tf.int32, [None, None], name='inputs_mask')
        
        dc_seq = tf.placeholder(tf.int32, [None, None], name='dc_seq')  # id in vocab
        dc_mask = tf.placeholder(tf.int32, [None, None], name='dc_mask')

        labels_seq = tf.placeholder(tf.int32, [None, None], name='labels_seq')  # id in vocab
        labels_mask = tf.placeholder(tf.int32, [None, None], name='labels_mask')
        
        # input sequence: could not prefix and suffix, when preparing examples
        # label sequence: suffix with a [end] token, then do [pad].
        #
        # decoder input seq: prefix with [start], suffix with [end], then do [pad].
        #
        print(inputs_seq)
        #
        input_tensors = (inputs_seq, inputs_mask, dc_seq, dc_mask)
        label_tensors = (labels_seq, labels_mask)
        #
        return input_tensors, label_tensors
    
    @staticmethod
    def build_inference(settings, input_tensors):
        
        inputs_seq, inputs_mask, dc_seq, dc_mask = input_tensors
        
        #
        dim_all = settings.num_heads * settings.num_units
        #
        keep_prob = tf.get_variable("keep_prob", shape=[], dtype=tf.float32, trainable=False)
        #
        with tf.device('/cpu:0'):
            emb_mat = tf.get_variable('token_embeddings',
                                      [settings.vocab.size(), settings.vocab.emb_dim],
                                      initializer=tf.constant_initializer(settings.vocab.embeddings),
                                      trainable = settings.emb_tune)
        #
        pe_mat = calculate_position_emb_mat(settings.max_seq_len, settings.posi_emb_dim,
                                            settings.d_model, "posi_embeddings")
        #
        with tf.variable_scope("encoder_decoder"):

            att_args = (settings.num_head, settings.num_units, keep_prob)
            ffd_args = (dim_all, dim_all, keep_prob)
            src_args = (settings.num_head, settings.num_units, keep_prob)
            #
            emb_trans = lambda x: get_emb_positioned(x, emb_mat, pe_mat)
            
            encoder = Encoder(settings.num_layers, EncoderLayer,
                              (dim_all, att_args, ffd_args, keep_prob))
            decoder = Decoder(settings.num_layers, DecoderLayer,
                              (dim_all, att_args, src_args, ffd_args, keep_prob))
            
            model = EncoderDecoder(encoder, decoder, emb_trans, emb_trans,
                                   Generator(dim_all, settings.decoder_vocab_size))
            #
            # model vars are all defined by now
            # graph yet
            #
            
        #
        if settings.is_train:
            src_mask = get_mask_mat_from_mask_seq(inputs_mask)
            dcd_mask = get_mask_mat_from_mask_seq(dc_mask)
            out = model.forward(inputs_seq, src_mask, dc_seq, dcd_mask)
            
            logits = model.generator.forward(out)
            logits_normed = tf.nn.softmax(logits, name = 'logits')
            preds = tf.nn.argmax(logits, name="preds")
        else:
            src_mask = get_mask_mat_from_mask_seq(inputs_mask)
            
            if settings.beam_width == 1:
                logits, preds_d = model.do_greedy_decoding(inputs_seq, src_mask,
                                                           settings.max_len_decoding,
                                                           settings.start_symbol_id)
                logits_normed = tf.identity(logits, name = 'logits')
                preds = tf.identity(preds_d, name="preds")
            else:
                logits, preds_d = model.do_beam_search_decoding(inputs_seq, src_mask,
                                                                settings.max_len_decoding,
                                                                settings.start_symbol_id,
                                                                settings.beam_width)
                logits_normed = tf.identity(logits, name = 'logits')
                preds = tf.identity(preds_d, name="preds")
            #
        
        #
        print(logits_normed)
        print(preds)
        #
        output_tensors = logits_normed, logits
        #   
        return output_tensors
    
    @staticmethod
    def build_loss_and_metric(settings, output_tensors, label_tensors):
        
        normed_logits, logits = output_tensors
        labels_seq, labels_mask = label_tensors
        
        labels_len = tf.cast(tf.reduce_sum(labels_mask, -1), dtype=tf.float32)
        
        with tf.variable_scope('loss'):
            
            # cross entropy on vocab (extended vocab)
            cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels_seq,  # labels_ed
                                                                       logits = logits)
            loss_batch = tf.multiply(cross_ent, tf.cast(labels_mask, dtype=tf.float32))
            loss_batch = tf.reduce_sum(loss_batch, -1)
            loss_batch = tf.divide(loss_batch, labels_len, name='loss_batch')
            loss = tf.reduce_mean(loss_batch, axis = 0, name = 'loss')
        
        with tf.variable_scope('metric'):
            
            metric = tf.constant(0.1, name = 'metric')
            
        #
        print(loss)
        print(metric)
        #
        return loss, metric
        #

class LabelSmooth():
    """
    """
    def __init__(self):
        pass
    
    def __call__(self):
        
        pass
    
    
        