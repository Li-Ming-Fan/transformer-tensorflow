# -*- coding: utf-8 -*-

from Zeras.layers import Dense, LayerNorm
from Zeras.layers import MultiHeadAttention, PositionwiseFeedForward
from Zeras.layers import SublayerWrapper
from Zeras.layers import build_module_copies

import decoding_method as dm


class ModelEncoderDecoder():
    """ assembling class
    """
    def __init__(self, encoder, decoder, generator):
        """
        """
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        
    def forward(self, src, src_mask, dcd_inputs, dcd_mask, crs_mask):
        """ source, decode, cross
        """
        return self.decode(dcd_inputs, dcd_mask, self.encode(src, src_mask), crs_mask)
        
    def encode(self, src, src_mask):
        """ src: source sequence token-ids
        """
        return self.encoder.forward(src, src_mask)
    
    def decode(self, dcd_inputs, dcd_mask, memory, crs_mask):
        """ decode one-step
        """
        return self.decoder.forward(dcd_inputs, dcd_mask, memory, crs_mask)
    
    def do_greedy_decoding(self, src, src_mask, max_len,
                           subs_masks, dcd_crs_masks, start_symbol_id):
        """ decode max_len steps
        """
        return dm.do_greedy_decoding(self, src, src_mask, max_len, 
                                     subs_masks, dcd_crs_masks, start_symbol_id)
    
    def do_beam_search_decoding(self, src, src_mask, max_len,
                                subs_masks, dcd_crs_masks, start_symbol_id):
        """ decode max_len steps
        """
        return dm.do_beam_search_decoding(self, src, src_mask, max_len,
                                          subs_masks, dcd_crs_masks, start_symbol_id)
    
#   
class Generator():
    """ project features to vocab
    """
    def __init__(self, d_features, vocab_size, emb_mat=None, scope="proj"):
        """
        """        
        self.emb_mat = emb_mat
        self.proj = Dense(d_features, vocab_size, weight_mat=emb_mat, scope=scope)
    
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
    def __init__(self, emb_trans_idx2emb, keep_prob, settings, scope="encoder"):
        """
        """
        self.emb_trans = emb_trans_idx2emb
        
        # norm_start
        self.layer_norm_start = LayerNorm(settings.dim_all, scope=scope)

        # layer & drop & add & norm
        att_args = settings.num_heads, settings.num_units, keep_prob
        ffd_args = settings.dim_all, settings.dim_ffm, keep_prob
        module_args = settings.dim_all, att_args, ffd_args, keep_prob
        self.layers = build_module_copies(EncoderLayer, module_args,
                                          settings.num_layers, scope = scope)

    # def __call__(self, x, mask):
    def forward(self, x, mask):
        """ x: seq_ids
        """
        # embedding
        x = self.emb_trans(x)
        # norm
        x = self.layer_norm_start(x)
        # layers
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
class EncoderLayer():
    """
    """
    def __init__(self, num_dim_all, att_args, ffd_args, keep_prob):
        """
        """
        self.sublayer_0 = SublayerWrapper(MultiHeadAttention, att_args,
                                          num_dim_all, keep_prob,
                                          scope = "sublayer_0")
        self.sublayer_1 = SublayerWrapper(PositionwiseFeedForward, ffd_args,
                                          num_dim_all, keep_prob,
                                          scope = "sublayer_1")
        
    def __call__(self, x, mask):
        """
        """        
        x = self.sublayer_0(x, lambda x: self.sublayer_0.sublayer(x, x, x, mask))
        x = self.sublayer_1(x, self.sublayer_1.sublayer)
        return x

#
class Decoder():
    """
    """
    def __init__(self, emb_trans_idx2emb, keep_prob, settings, scope="decoder"):
        """
        """
        self.emb_trans = emb_trans_idx2emb
        
        # norm_start
        self.layer_norm_start = LayerNorm(settings.dim_all, scope=scope)
        
        # layer & drop & add & norm
        att_args = settings.num_heads, settings.num_units, keep_prob
        crs_args = settings.num_heads, settings.num_units, keep_prob
        ffd_args = settings.dim_all, settings.dim_ffm, keep_prob
        module_args = settings.dim_all, att_args, crs_args, ffd_args, keep_prob
        self.layers = build_module_copies(DecoderLayer, module_args,
                                          settings.num_layers, scope = scope)
    
    def forward(self, x, dcd_mask, memory, crs_mask):
        """ x: seq_ids
        """
        # embedding
        x = self.emb_trans(x)
        # norm
        x = self.layer_norm_start(x)
        # layers
        for layer in self.layers:
            x = layer(x, dcd_mask, memory, crs_mask)        
        return x

class DecoderLayer():
    """
    """
    def __init__(self, num_dim_all, att_args, crs_args, ffd_args, keep_prob):
        """
        """
        self.sublayer_0 = SublayerWrapper(MultiHeadAttention, att_args,
                                          num_dim_all, keep_prob,
                                          scope = "sublayer_0")
        self.sublayer_1 = SublayerWrapper(MultiHeadAttention, crs_args,
                                          num_dim_all, keep_prob,
                                          scope = "sublayer_1")
        self.sublayer_2 = SublayerWrapper(PositionwiseFeedForward, ffd_args,
                                          num_dim_all, keep_prob,
                                          scope = "sublayer_2")
        
    def __call__(self, x, dcd_mask, memory, crs_mask):
        """
        """        
        m = memory
        x = self.sublayer_0(x, lambda x: self.sublayer_0.sublayer(x, x, x, dcd_mask))
        x = self.sublayer_1(x, lambda x: self.sublayer_1.sublayer(x, m, m, crs_mask))
        x = self.sublayer_2(x, self.sublayer_2.sublayer)
        return x
        
#

    