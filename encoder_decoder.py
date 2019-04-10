# -*- coding: utf-8 -*-

from zoo_layers import Dense, LayerNorm
from zoo_layers import MultiHeadAttention, PositionwiseFeedForward
from zoo_layers import SublayerWrapper
from zoo_layers import build_module_copies

import decoding_method as dm


class EncoderDecoder():
    """
    """
    def __init__(self, encoder, decoder,
                 src_emb_trans, tgt_emb_trans, generator):
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb_trans = src_emb_trans
        self.tgt_emb_trans = tgt_emb_trans
        self.generator = generator
        
    def forward(self, src, src_mask, dc_inputs, dc_mask):        
        return self.decode(self.encode(src, src_mask), src_mask, dc_inputs, dc_mask)
        
    def encode(self, src, src_mask):
        return self.encoder(self.src_emb_trans(src), src_mask)
    
    def decode(self, memory, src_mask, dc_inputs, dc_mask):
        """ decode one-step
        """
        return self.decoder(memory, src_mask, self.tgt_emb_trans(dc_inputs), dc_mask)
    
    def do_greedy_decoding(self, src, src_mask, max_len, subs_masks, start_symbol_id):
        """ decode max_len steps
        """
        return dm.do_greedy_decoding(self, src, src_mask, max_len, subs_masks, start_symbol_id)
    
    def do_beam_search_decoding(self, src, src_mask, max_len, subs_masks, start_symbol_id):
        """ decode max_len steps
        """
        return dm.do_beam_search_decoding(self, src, src_mask, max_len, subs_masks, start_symbol_id)
    
#   
class Generator():
    """
    """
    def __init__(self, d_model, vocab_size, emb_mat=None, scope="proj"):
        """
        """        
        self.emb_mat = emb_mat
        self.proj = Dense(d_model, vocab_size, weight_mat=emb_mat, scope=scope)
    
    def forward(self, x):
        if self.emb_mat is None:
            out = self.proj(x, transpose_b=False)
        else:
            out = self.proj(x, transpose_b=True)
        return out
        
#  
class Encoder():
    """
    """
    def __init__(self, num_layers, layer_module, module_args, scope="encoder"):
        
        self.layers = build_module_copies(layer_module, module_args, num_layers,
                                          scope = scope)
        self.layer_norm = LayerNorm(layer_module.num_all, scope=scope)
    
    def __call__(self, x, mask):
        """ x: seq_embedded
        """        
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)

class Decoder():
    """
    """
    def __init__(self, num_layers, layer_module, module_args, scope="encoder"):
        
        self.layers = build_module_copies(layer_module, module_args, num_layers,
                                          scope = scope)
        self.layer_norm = LayerNorm(layer_module.num_all, scope=scope)
    
    def __call__(self, memory, src_mask, x, dc_mask):
        """ x: seq_embedded
        """        
        for layer in self.layers:
            x = layer(memory, src_mask, x, dc_mask)
        return self.layer_norm(x)
    
#
class EncoderLayer():
    """
    """
    def __init__(self, num_dim_all, att_args, ffd_args, keep_prob):
        
        # self.num_dim_all = num_dim_all
        self.sublayer_0 = SublayerWrapper(num_dim_all, keep_prob,
                                          MultiHeadAttention, att_args,
                                          scope = "sublayer_0")
        self.sublayer_1 = SublayerWrapper(num_dim_all, keep_prob,
                                          PositionwiseFeedForward, ffd_args,
                                          scope = "sublayer_1")
        
    def __call__(self, x, mask):
        
        x = self.sublayer_0(x, lambda x: self.sublayer_0.sublayer(x, x, x, mask))
        x = self.sublayer_1(x, self.sublayer_1.sublayer)
        return x
    
class DecoderLayer():
    """
    """
    def __init__(self, num_dim_all, att_args, src_args, ffd_args, keep_prob):
        
        # self.num_dim_all = num_dim_all
        self.sublayer_0 = SublayerWrapper(num_dim_all, keep_prob,
                                          MultiHeadAttention, att_args,
                                          scope = "sublayer_0")
        self.sublayer_1 = SublayerWrapper(num_dim_all, keep_prob,
                                          MultiHeadAttention, src_args,
                                          scope = "sublayer_1")
        self.sublayer_2 = SublayerWrapper(num_dim_all, keep_prob,
                                          PositionwiseFeedForward, ffd_args,
                                          scope = "sublayer_2")
        
    def __call__(self, memory, src_mask, x, dc_mask):
        
        m = memory
        x = self.sublayer_0(x, lambda x: self.sublayer_0.sublayer(x, x, x, dc_mask))
        x = self.sublayer_1(x, lambda x: self.sublayer_1.sublayer(x, m, m, src_mask))
        x = self.sublayer_2(x, self.sublayer_2.sublayer)
        return x
        
#

    