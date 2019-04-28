# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:53:37 2019

@author: limingfan
"""

import os
import random
import numpy as np

from Zeras.vocab import Vocab

#
def get_examples_generator(data_files = []):
    """
    """
    def examples_generator(single_pass=True):
        """
        """
        vocab = Vocab()
        for idx in range(10):
            vocab.add(str(idx))
        #
        count_examples = 0
        while True:
            src_seq_token = [str(random.randint(0, 9)) for _ in range(random.randint(1, 10))]
            src_seq = vocab.convert_tokens_to_ids(src_seq_token)
            
            if single_pass and count_examples == 600: break
            count_examples += 1
        
            yield src_seq
            #
    #
    return examples_generator
    #

def batch_std_transor(list_examples, max_seq_len = 20, tgt_seq_len = 12):
    """
    """    
    start_id = 2
    end_id = 3
    #
    num_examples = len(list_examples)
    list_len = [len(item) for item in list_examples]
    max_len = min(max_seq_len, max(list_len))
    
    src_seq = np.zeros((num_examples, max_len), dtype=np.int32)
    src_seq_mask = np.zeros((num_examples, max_len), dtype=np.int32)
    
    dcd_seq = np.zeros((num_examples, tgt_seq_len), dtype=np.int32)
    dcd_seq_mask = np.zeros((num_examples, tgt_seq_len), dtype=np.int32)
    lbl_seq = np.zeros((num_examples, tgt_seq_len), dtype=np.int32)
    lbl_seq_mask = np.zeros((num_examples, tgt_seq_len), dtype=np.int32)
    
    for eid in range(num_examples):
        
        dcd_seq[eid, 0] = start_id
        dcd_seq_mask[eid, 0] = 1
        
        src_len = list_len[eid]        
        for tid in range(src_len):
            src_seq[eid, tid] = list_examples[eid][tid]
            src_seq_mask[eid, tid] = 1
            
            if tid >= tgt_seq_len: continue
            lbl_seq[eid, tid] = list_examples[eid][tid]
            lbl_seq_mask[eid, tid] = 1
            
            if tid+1 >= tgt_seq_len: continue
            dcd_seq[eid, tid+1] = list_examples[eid][tid]
            dcd_seq_mask[eid, tid+1] = 1

        #
        if src_len >= tgt_seq_len: continue
        lbl_seq[eid, src_len] = end_id
        lbl_seq_mask[eid, src_len] = 1
        
        if src_len+1 >= tgt_seq_len: continue
        dcd_seq[eid, src_len+1] = end_id
        dcd_seq_mask[eid, src_len+1] = 1
        #
        
    #
    return src_seq, src_seq_mask, dcd_seq, dcd_seq_mask, lbl_seq, lbl_seq_mask
    #
    
#
if __name__ == "__main__":
    
    #
    vocab = Vocab()
    for idx in range(10):
        vocab.add(str(idx))
    
    dir_vocab = "./vocab"
    if not os.path.exists(dir_vocab): os.mkdir(dir_vocab)
    token_file = os.path.join(dir_vocab, "vocab_tokens.txt")
    vocab.save_tokens_to_file(token_file)
    
    #
    data_gen = get_examples_generator()
    it = data_gen()
    print(next(it))
    print(next(it))
    
    a = [next(it) for _ in range(3)]
    print(a)
    
    batch = batch_std_transor(a)
    print(batch[0])
    print(batch[1])
    print(batch[2])
    print(batch[3])
    print(batch[4])
    print(batch[5])