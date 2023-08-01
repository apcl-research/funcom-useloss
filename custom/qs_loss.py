import sys
import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
import numpy as np
import tensorflow_hub as tfhub
from tensorflow import keras
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.losses import cosine_similarity, categorical_crossentropy, KLDivergence, Reduction
from tensorflow.python.keras import backend 
from tensorflow.keras import layers

import numpy as np
import parallel_sort

sys.path.append(os.path.abspath('../'))
import tokenizer
from myutils import index2word



module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
model = tfhub.load(module_url)


def use_prep(comstok):
	index_tensor = tf.Variable(list(comstok.i2w.keys()), dtype=tf.int64)
	comwords = list(comstok.i2w.values())
	comwords_tensor = tf.Variable(comwords)
	return index_tensor, comwords_tensor



# Basic cce loss
def custom_cce_loss():
    def qs_cce_loss(y_true, y_pred):
        cce = keras.losses.categorical_crossentropy(y_true, y_pred)
        return cce
    return qs_cce_loss


# Implementation of Bleu loss proposed by Wieting et al.
# https://arxiv.org/abs/1909.06694

def custom_bleu_base_loss(index_tensor, comwords_tensor):
    def qs_cce_loss(y_true, y_pred):
        comlen = 13 # length of output comment


        loss = keras.losses.categorical_crossentropy(y_true, y_pred)

        y_true_arg = tf.argmax(y_true, axis=1)
        y_pred_arg = tf.argmax(y_pred, axis=1)
        i2wtable = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(index_tensor, comwords_tensor), default_value='<UNK>')


        y_true_comwords = i2wtable.lookup(y_true_arg)
        y_pred_comwords = i2wtable.lookup(y_pred_arg)


        y_true_comwords_sen = []
        y_pred_comwords_sen = []
        y_true_comwords_s = y_true_comwords.numpy()
        y_pred_comwords_s = y_pred_comwords.numpy()

        for i in range(0, y_true.shape[0], comlen):
            temp1 = ''
            temp2 = ''
            for j in range(i, i+comlen):
                temp1 += ' ' + tf.compat.as_str_any(y_true_comwords_s[j])
                temp2 += ' ' + tf.compat.as_str_any(y_pred_comwords_s[j])
            temp1 = temp1.strip()
            temp2 = temp2.strip()
            y_true_comwords_sen.append([temp1])
            y_pred_comwords_sen.append(temp2)

        y_true_comwords_sen = np.asarray(y_true_comwords_sen)
        y_pred_comwords_sen = np.asarray(y_pred_comwords_sen)

        bleu_score = corpus_bleu(y_true_comwords_sen,y_pred_comwords_sen)
       
                

        loss = loss + bleu_score
        

        loss = tf.reduce_mean(loss)
        

        return loss
 

    return qs_cce_loss


# Implementation of Simile loss proposed by Wieting et al.
# https://arxiv.org/abs/1909.06694


def custom_use_simile_loss(index_tensor, comwords_tensor):
    def qs_cce_loss(y_true, y_pred):
        comlen = 13 # length of output comment
        loss = keras.losses.categorical_crossentropy(y_true, y_pred)

        y_true_arg = tf.argmax(y_true, axis=1)
        y_pred_arg = tf.argmax(y_pred, axis=1)
        i2wtable = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(index_tensor, comwords_tensor), default_value='<UNK>')


        y_true_comwords = i2wtable.lookup(y_true_arg)
        y_pred_comwords = i2wtable.lookup(y_pred_arg)


        y_true_comwords_sen = []
        y_pred_comwords_sen = []
        y_true_comwords_s = y_true_comwords.numpy()
        y_pred_comwords_s = y_pred_comwords.numpy()

        for i in range(0, y_true.shape[0], comlen):
            temp1 = ''
            temp2 = ''
            for j in range(i, i+comlen):
                temp1 += ' ' + tf.compat.as_str_any(y_true_comwords_s[j])
                temp2 += ' ' + tf.compat.as_str_any(y_pred_comwords_s[j])
            temp1 = temp1.strip()
            temp2 = temp2.strip()
            y_true_comwords_sen.append(temp1)
            y_pred_comwords_sen.append(temp2)

        lps = list()


        for i in range(0, len(y_true_comwords_sen)):
            
            y_true_sen = y_true_comwords_sen[i].split(' ')
            y_pred_sen = y_pred_comwords_sen[i].split(' ')


            try:
                l_ref = y_true_sen.index('</s>')
            except:
                l_ref = comlen

            try:
                l_prd = y_pred_sen.index('</s>')
            except:
                l_prd = comlen

            lp = math.exp(1-(max(l_ref, l_prd) / min(l_ref, l_prd))) # lp term in simile paper


            for j in range(0, comlen):
                lps.append(lp)

        lps = np.asarray(lps, dtype=np.float32)


        y_true_comwords_use = model(y_true_comwords_sen)
        y_pred_comwords_use = model(y_pred_comwords_sen)
        
        use_score = tf.keras.losses.cosine_similarity(y_true_comwords_use, y_pred_comwords_use)
        use_score = use_score * (-1)
        use_score = tf.expand_dims(use_score, axis=1)
        use_score = tf.repeat(use_score, repeats=[comlen], axis=1)
        
        use_score = tf.reshape(use_score, [y_true_arg.shape[0]])
        

        loss = loss + use_score

        loss = (loss + 1) / 2



        loss = 0.25*lps + 0.75*loss


        loss = tf.reduce_mean(loss)


        return loss
 

    return qs_cce_loss



def custom_use_seq(index_tensor, comwords_tensor):
    def qs_cce_loss(y_true, y_pred):
        beta = 0.8 # beta value for the adjustment of exponential function
        comlen = 13 # length of output comment


        loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        
        y_true_arg = tf.argmax(y_true, axis=1)
        y_pred_arg = tf.argmax(y_pred, axis=1)
        i2wtable = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(index_tensor, comwords_tensor), default_value='<UNK>')


        y_true_comwords = i2wtable.lookup(y_true_arg)
        y_pred_comwords = i2wtable.lookup(y_pred_arg)


        y_true_comwords_sen = []
        y_pred_comwords_sen = []
        y_true_comwords_s = y_true_comwords.numpy()
        y_pred_comwords_s = y_pred_comwords.numpy()


        for i in range(0, y_true.shape[0], comlen):
            temp1 = ''
            temp2 = ''
            for j in range(i, i+comlen):
                temp1 += ' ' + tf.compat.as_str_any(y_true_comwords_s[j])
                temp2 += ' ' + tf.compat.as_str_any(y_pred_comwords_s[j])
            temp1 = temp1.strip()
            temp2 = temp2.strip()
            y_true_comwords_sen.append(temp1)
            y_pred_comwords_sen.append(temp2)

        
        y_true_comwords_use = model(y_true_comwords_sen)
        y_pred_comwords_use = model(y_pred_comwords_sen)
        
        use_score = tf.keras.losses.cosine_similarity(y_true_comwords_use, y_pred_comwords_use)
        
        # reverse because the output of tf.keras.losses.cosine_similarity() will be 1 if two inputs are totally opposite and vice versa.
        use_score = use_score * (-1)
        use_score = tf.expand_dims(use_score, axis=1)
        use_score = tf.repeat(use_score, repeats=[comlen], axis=1)
        
        use_score = tf.reshape(use_score, [y_true_arg.shape[0]])
        mask = tf.where(use_score >= 0, tf.where(y_true_arg == y_pred_arg, 1.0, 0.), tf.where(y_true_arg == y_pred_arg, 0., 1.0))
        use_score = mask * use_score
        
        all_weight = tf.math.exp(use_score/beta)

        loss = loss * all_weight

        loss = tf.reduce_mean(loss)


        return loss

    return qs_cce_loss



