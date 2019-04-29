# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 21:12:23 2018

@author: limingfan
"""

from Zeras.model_settings_baseboard import ModelSettingsBaseboard


class ModelSettings(ModelSettingsBaseboard):
    """
    """
    def __init__(self):
        """
        """
        super(ModelSettings, self).__init__()
        
        # task
        self.task = "copy"
        self.tokens_file = "./vocab/vocab_tokens.txt"
        self.emb_file = None
        
        # model
        self.model_tag = None
        self.is_train = None

        # data macro
        self.min_seq_len = 1     #
        self.max_seq_len = 12    #
        
        # vocab
        self.emb_dim = 128
        self.emb_tune = 1  # 1 for tune, 0 for not
        self.posi_emb_dim = self.emb_dim

        # model macro
        self.num_layers = 6
        self.num_heads = 8
        self.num_units = int(self.emb_dim / self.num_heads)
        self.dim_all = self.num_heads * self.num_units
        self.dim_model = self.emb_dim
        self.dim_ffm = 256
        #
        
        # self.decoder_vocab_size = 17
        self.beam_width = 1
        self.max_len_decoding = self.max_seq_len
        self.start_symbol_id = 2  #
        
        #
        # train
        self.gpu_available = "0"  # specified in args
        self.gpu_batch_split = [16, 20]   # list; if None, batch split evenly
        #
        self.gpu_mem_growth = True
        self.log_device = False
        self.soft_placement = True
        
        self.with_bucket = False
        
        #
        self.num_epochs = 100
        self.batch_size = 36
        self.batch_size_eval = 6
        self.max_batches_eval = 20
        
        self.reg_lambda = 0.0  # 0.0, 0.01
        self.grad_clip = 0.0  # 0.0, 5.0, 8.0, 2.0
        self.keep_prob = 0.9  # 1.0, 0.7, 0.5
        self.label_smoothing = 0.01
        
        self.optimizer_type = 'adam'  # adam, momentum, sgd, customized
        self.momentum = 0.9
        self.learning_rate_base = 0.001   #
        self.learning_rate_minimum = 0.000001
        self.warmup_steps = 1000
        self.decay_steps = 5000
        self.decay_rate = 0.99
        self.staircase = True
        
        self.check_period_batch = 100
        self.valid_period_batch = 100
        #

        # inputs/outputs
        self.vs_str_multi_gpu = "vs_multi_gpu"
        #
        self.inputs_predict_name = ['src_seq:0', 'src_seq_mask:0']
        self.outputs_predict_name = ['vs_multi_gpu/logits:0']
        self.pb_outputs_name = ['vs_multi_gpu/logits']
                
        self.inputs_train_name = ['src_seq:0', 'src_seq_mask:0',
                                  'dcd_seq:0', 'dcd_seq_mask:0',
                                  'labels_seq:0', 'labels_mask:0']
        self.outputs_train_name = ['vs_multi_gpu/logits:0']
        self.use_metric = True
        
        self.debug_tensors_name = ['vs_multi_gpu/loss/loss:0',
                                   'vs_multi_gpu/logits:0',
                                   'vs_multi_gpu/preds:0'
                                   #'encoder/encoder_rnn_1/bw/bw/sequence_length:0',
                                   #'inputs_len:0'
                                   ]
        
        #
        self.base_dir = './task_copy_results'
        # self.model_dir = None   # if not set, default values will be used.
        # self.model_name = None
        # self.pb_file = None
        # self.log_dir = None
        # self.log_path = None
        #
   
#      
if __name__ == "__main__":
    
    sett = ModelSettings()
    #
    sett.model_tag = 'cnn'
    sett.is_train = False
    #
    sett.check_settings()
    #
    
    #
    info_dict = sett.trans_info_to_dict()
    print("original:")
    print(info_dict)
    print()
    #
    info_dict["model_tag"] = "transformer"
    sett.assign_info_from_dict(info_dict)
    #
    info_dict = sett.trans_info_to_dict()
    print("assigned:")
    print(info_dict)
    print()
    #
    
    #
    file_path = "./temp_settings.json"
    sett.save_to_json_file(file_path)    
    sett.load_from_json_file(file_path)
    #
    info_dict = sett.trans_info_to_dict()
    print("saved then loaded:")
    print(info_dict)
    print()
    #

    #    
    sett.close_logger()
    #