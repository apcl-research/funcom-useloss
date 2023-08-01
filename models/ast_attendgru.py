from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, GRU, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, RepeatVector, Permute, TimeDistributed, dot
from tensorflow.keras.optimizers import RMSprop, Adamax
import tensorflow.keras as keras
import tensorflow.keras.utils
import tensorflow as tf
from tensorflow.keras import metrics


# Thanks LeClair et al. for providing the open source implementation of their model.
# https://arxiv.org/abs/1902.01954
# https://github.com/mcmillco/funcom

class AstAttentionGRUModel:
    def __init__(self, config):
        
        config['tdatlen'] = 50
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.smlvocabsize = config['smlvocabsize']
        self.tdatlen = config['tdatlen']
        self.comlen = config['comlen']
        self.smllen = config['smllen']
        
        self.embdims = 100
        self.smldims = 100
        self.recdims = 100
        self.findims = 100

        self.config['batch_config'] = [ ['tdat', 'com', 'smlseq'], ['comout'] ]

    def create_model(self):
        
        dat_input = Input(shape=(self.tdatlen,))
        com_input = Input(shape=(self.comlen,))
        sml_input = Input(shape=(self.smllen,))
        
        ee = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)(dat_input)
        se = Embedding(output_dim=self.smldims, input_dim=self.smlvocabsize, mask_zero=False)(sml_input)

        se_enc = GRU(self.recdims, return_state=True, return_sequences=True)
        seout, state_sml = se_enc(se)

        enc = GRU(self.recdims, return_state=True, return_sequences=True)
        encout, state_h = enc(ee, initial_state=state_sml)

        de = Embedding(output_dim=self.embdims, input_dim=self.comvocabsize, mask_zero=False)(com_input)
        dec = GRU(self.recdims, return_sequences=True)
        decout = dec(de, initial_state=state_h)

        attn = dot([decout, encout], axes=[2, 2])
        attn = Activation('softmax')(attn)

        ast_attn = dot([decout, seout], axes=[2, 2])
        ast_attn = Activation('softmax')(ast_attn)

        context = dot([attn, encout], axes=[2, 1])
        ast_context = dot([ast_attn, seout], axes=[2, 1])

        context = concatenate([context, decout, ast_context])

        out = TimeDistributed(Dense(self.findims, activation="relu"))(context)

        out = Flatten()(out)
        out = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[dat_input, com_input, sml_input], outputs=out)
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.config, model
