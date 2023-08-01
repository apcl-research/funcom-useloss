import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, GRU, Concatenate, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, GlobalAveragePooling1D
from tensorflow.compat.v1.keras.layers import Bidirectional, RepeatVector, Permute, TimeDistributed, dot
from tensorflow.keras.optimizers import RMSprop, Adamax
from tensorflow.keras import utils, metrics

from custom.qstransformer_layers import TransformerBlock, TokenAndPositionEmbedding, MultiHeadAttentionBlock
from custom.qs_loss import custom_cce_loss

# setransformer baseline of IEEE Transactions on Reliability Li et al.
# https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=24

class SeTransformer:
    def __init__(self, config):
        
        config['tdatlen'] = 50
        config['smllen'] = 50
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.smlvocabsize = config['smlvocabsize']
        self.datlen = config['tdatlen']
        self.comlen = config['comlen']
        self.smllen = config['smllen']
        
        self.embdims = 100
        self.smldims = 100
        self.attheads = 2 # number of attention heads
        self.recdims = 100 
        self.ffdims = 100 # hidden layer size in feed forward network inside transformer
        self.kernel = 3 # cnn kernel size
        self.pool_size = 2 # cnn pool size
        self.strides = 1 # cnn pool layere stride size
        self.padding = False # if pad to the same size as input after convolution and pooling

        self.config['batch_config'] = [ ['tdat', 'com', 'smlseq'], ['comout'] ]
        self.config['loss_type'] = config['loss_type']

    def create_model(self):
        
        dat_input = Input(shape=(self.datlen,))
        com_input = Input(shape=(self.comlen,))
        sml_input = Input(shape=(self.smllen,))

        ee = TokenAndPositionEmbedding(self.datlen, self.tdatvocabsize, self.embdims)
        eeout = ee(dat_input)
        if(not self.padding):
            eeout = Conv1D(self.embdims, self.kernel, activation='tanh',input_shape=(None, ))(eeout)
            eeout = MaxPooling1D(pool_size=self.pool_size, strides=self.strides, padding='valid')(eeout)
        else:
            eeout = Conv1D(self.embdims, self.kernel, activation='tanh',input_shape=(None, self.embdims), padding ='same')(eeout)
            eeout = MaxPooling1D(pool_size=self.pool_size, strides=self.strides, padding='same')(eeout)
        eeout = Dense(self.embdims, activation="tanh")(eeout)

        se = TokenAndPositionEmbedding(self.smllen, self.smlvocabsize, self.embdims)
        seout = se(sml_input)
        if(not self.padding):
            seconv = Conv1D(self.embdims, self.kernel, activation='tanh',input_shape=(None, ))
            seconvout = seconv(seout)
            seout = MaxPooling1D(pool_size=self.pool_size, strides=self.strides, padding='valid')(seconvout)
        else:
            seconv = Conv1D(self.embdims, self.kernel, activation='tanh',input_shape=(None, self.embdims), padding='same')
            seconvout = seconv(seout)
            seout = MaxPooling1D(pool_size=self.pool_size, strides=self.strides, padding='same')(seconvout)
        seout = Dense(self.embdims, activation="tanh")(seout)

        etransformer_block = TransformerBlock(self.embdims, self.attheads, self.ffdims)
        encout = etransformer_block(eeout, eeout)
        setransformer_block = TransformerBlock(self.embdims, self.attheads, self.ffdims)
        sencout = setransformer_block(seout, seout)

        de = TokenAndPositionEmbedding(self.comlen, self.comvocabsize, self.embdims)
        deout = de(com_input)
        de_mha1 = MultiHeadAttentionBlock(self.embdims, self.attheads)
        de_mha2 = MultiHeadAttentionBlock(self.embdims, self.attheads)
        ee_de_mha1_out = de_mha1(encout, deout)
        se_de_mha1_out = de_mha2(sencout, deout)


        dtransformer_block = TransformerBlock(self.embdims, self.attheads, self.ffdims)
        decout = dtransformer_block(se_de_mha1_out, ee_de_mha1_out)

        #decout = dtransformer_block(eeout, eeout)
        context = decout
        out = context
        # out = TimeDistributed(Dense(self.recdims, activation="tanh"))(context)
        out = Flatten()(out)
        out = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[dat_input, com_input, sml_input], outputs=out)
        lossf = custom_cce_loss()
        if self.config['loss_type'] == 'use':
            lossf = custom_use_loss(self.index_tensor, self.comwords_tensor)
        elif self.config['loss_type'] == 'attendgru':
            lossf = custom_attendgru_loss(self.fmodel)
        elif self.config['loss_type'] == 'use-dist':
            lossf = custom_dist_cce_loss(self.dist)

        model.compile(loss=lossf, optimizer='adam', metrics=['accuracy'])
        return self.config, model
