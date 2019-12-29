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
        self.activation = "gelu"
        #
        self.use_metric_in_graph = True
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

        #
        self.num_epochs = 100
        self.batch_size = 36
        self.batch_size_eval = 6
        self.max_batches_eval = 20
        
        self.reg_lambda = 0.0  # 0.0, 0.01
        self.grad_clip = 0.0  # 0.0, 5.0, 8.0, 2.0
        self.keep_prob = 0.9  # 1.0, 0.7, 0.5
        self.label_smoothing = 0.01
        
        self.optimizer_type = 'adam_wd'  # adam_wd, adam, momentum, sgd, customized
        self.beta_1 = 0.9
        self.learning_rate_base = 0.001   #
        self.learning_rate_minimum = 0.000001
        self.warmup_steps = 1000
        self.decay_steps = 5000
        self.lr_power = 1
        self.lr_cycle = True
        
        self.check_period_batch = 100
        self.valid_period_batch = 100
        #

        #
        self.base_dir = './task_copy_results'
        # self.model_dir = None   # if not set, default values will be used.
        # self.model_name = None
        # self.model_dir_best = None  
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