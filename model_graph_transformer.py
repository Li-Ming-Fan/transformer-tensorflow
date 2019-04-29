# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 07:18:29 2019

@author: limingfan
"""

import tensorflow as tf

from Zeras.nn import get_position_emb_mat, get_emb_positioned
from Zeras.nn import get_tensor_expanded
from Zeras.nn import get_mask_mat_subsequent
from Zeras.nn import get_label_smoothened

from model_encoder_decoder import ModelEncoderDecoder, Generator
from model_encoder_decoder import Encoder, Decoder


class ModelGraph():
    
    @staticmethod
    def build_placeholder(settings):
        
        src_seq = tf.placeholder(tf.int32, [None, None], name='src_seq')  # id in vocab
        src_seq_mask = tf.placeholder(tf.int32, [None, None], name='src_seq_mask')
        
        dcd_seq = tf.placeholder(tf.int32, [None, None], name='dcd_seq')  # id in vocab
        dcd_seq_mask = tf.placeholder(tf.int32, [None, None], name='dcd_seq_mask')

        labels_seq = tf.placeholder(tf.int32, [None, None], name='labels_seq')  # id in vocab
        labels_mask = tf.placeholder(tf.int32, [None, None], name='labels_mask')
        
        # input sequence: could not prefix and suffix, when preparing examples
        # label sequence: suffix with a [end] token, then do [pad].
        #
        # decoder input seq: prefix with [start], suffix with [end], then do [pad].
        #
        print(src_seq)
        #
        input_tensors = (src_seq, src_seq_mask, dcd_seq, dcd_seq_mask)
        label_tensors = (labels_seq, labels_mask)
        #
        return input_tensors, label_tensors
    
    @staticmethod
    def build_inference(settings, input_tensors):
        
        src_seq, src_seq_mask, dcd_seq, dcd_seq_mask = input_tensors
        
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
        pe_mat = get_position_emb_mat(settings.max_seq_len, settings.posi_emb_dim,
                                      settings.dim_model, "posi_embeddings")
        #
        with tf.variable_scope("encoder_decoder"):

            emb_trans = lambda x: get_emb_positioned(x, emb_mat, pe_mat)
            
            encoder = Encoder(emb_trans, keep_prob, settings)
            decoder = Decoder(emb_trans, keep_prob, settings)
            generator = Generator(dim_all, settings.vocab.size(), emb_mat=emb_mat)
            
            model = ModelEncoderDecoder(encoder, decoder, generator)
            #
            # model vars are all defined by now
            # graph yet
            #
            
        #
        src_mask = get_tensor_expanded(src_seq_mask, 1, dtype=tf.float32)
        crs_mask = get_tensor_expanded(src_seq_mask, 1, dtype=tf.float32)
        dcd_mask = get_tensor_expanded(dcd_seq_mask, 1, dtype=tf.float32)
        sub_mask = get_mask_mat_subsequent(settings.max_len_decoding)
        dcd_mask = dcd_mask * sub_mask
        #
        if settings.is_train:
            out = model.forward(src_seq, src_mask, dcd_seq, dcd_mask, crs_mask)
            logits = model.generator.forward(out)
            logits_normed = tf.nn.softmax(logits, -1, name = 'logits')
            preds = tf.argmax(logits, -1, name="preds")
        else:
            if settings.beam_width == 1:
                logits, preds_d = model.do_greedy_decoding(src_seq, src_mask,
                                                           settings.max_len_decoding,
                                                           sub_mask, crs_mask,
                                                           settings.start_symbol_id)
                logits_normed = tf.identity(logits, name = 'logits')
                preds = tf.identity(preds_d, name="preds")
            else:
                logits, preds_d = model.do_beam_search_decoding(src_seq, src_mask,
                                                                settings.max_len_decoding,
                                                                sub_mask, crs_mask,
                                                                settings.start_symbol_id,
                                                                settings.beam_width)
                logits_normed = tf.identity(logits, name = 'logits')
                preds = tf.identity(preds_d, name="preds")
            #
        
        #
        print(logits_normed)
        print(preds)
        #
        output_tensors = logits_normed, logits, preds
        #   
        return output_tensors
    
    @staticmethod
    def build_loss_and_metric(settings, output_tensors, label_tensors):
        
        normed_logits, logits, preds = output_tensors
        labels_seq, labels_mask = label_tensors
        
        labels_len = tf.cast(tf.reduce_sum(labels_mask, -1), dtype=tf.float32)
        
        with tf.variable_scope('loss'):
            #
            onehot_labels = tf.one_hot(labels_seq, settings.vocab.size())
            labels_smooth = get_label_smoothened(onehot_labels, settings.vocab.size(),
                                                 settings.label_smoothing)
            #
            cross_ent = tf.nn.softmax_cross_entropy_with_logits(labels = labels_smooth,
                                                                logits = logits)
            loss_batch = tf.multiply(cross_ent, tf.cast(labels_mask, dtype=tf.float32))
            loss_batch = tf.reduce_sum(loss_batch, -1)
            loss_batch = tf.divide(loss_batch, labels_len, name='loss_batch')
            loss = tf.reduce_mean(loss_batch, axis = 0, name = 'loss')
        
        with tf.variable_scope('metric'):
            
            # correct_pred = tf.cast(tf.equal(labels_seq, preds), tf.int32)
            # acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = 'metric')
            
            metric = tf.constant(0.1, name = 'metric')
            
        #
        print(loss)
        print(metric)
        #
        return loss, metric
        #

        