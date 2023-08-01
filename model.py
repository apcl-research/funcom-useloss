import tensorflow.keras as keras
import tensorflow as tf

from models.ast_attendgru import AstAttentionGRUModel as ast_attendgru
from models.ast_attendgru_use_seq import AstAttentionGRUUSESeqModel as ast_attendgru_use_seq
from models.ast_attendgru_simile import AstAttentionGRUSimile as ast_attendgru_simile
from models.ast_attendgru_bleu_base import AstAttentionGRUBleuBase as ast_attendgru_bleu_base
from models.codegnngru import CodeGNNGRUModel as codegnngru
from models.codegnngru_simile import CodeGNNGRUModelSimile as codegnngru_simile
from models.codegnngru_bleu_base import CodeGNNGRUModelBleuBase as codegnngru_bleu_base
from models.codegnngru_use_seq import CodeGNNGRUUSESeqModel as codegnngru_use_seq
from models.transformer_base import TransformerBase as xformer_base
from models.transformer_base_bleu_base import TransformerBaseBleuBase as xformer_base_bleu_base
from models.transformer_base_use_seq import TransformerBaseUSESeq as xformer_base_use_seq
from models.transformer_base_simile import TransformerBaseSimile as xformer_base_simile
from models.setransformer import SeTransformer as sexformer
from models.setransformer_use_seq import SeTransformerUSESeq as sexformer_use_seq
from models.setransformer_bleu_base import SeTransformerBleuBase as sexformer_bleu_base
from models.setransformer_simile import SeTransformerSimile as sexformer_simile


def create_model(modeltype, config):
    mdl = None 
    if modeltype == 'ast-attendgru': 
        mdl = ast_attendgru(config)
    elif modeltype == 'ast-attendgru-bleu-base':
        mdl = ast_attendgru_bleu_base(config)
    elif modeltype == 'ast-attendgru-use-seq':
        mdl = ast_attendgru_use_seq(config)
    elif modeltype == 'ast-attendgru-simile':
        mdl = ast_attendgru_simile(config)
    elif modeltype == 'codegnngru':
        mdl = codegnngru(config)
    elif modeltype == 'codegnngru-simile':
        mdl = codegnngru_simile(config)
    elif modeltype == 'codegnngru-use-seq':
        mdl = codegnngru_use_seq(config)
    elif modeltype == 'codegnngru-bleu-base':
        mdl = codegnngru_bleu_base(config)
    elif modeltype == 'transformer-base':
        mdl = xformer_base(config)
    elif modeltype == 'transformer-base-bleu-base':
        mdl = xformer_base_bleu_base(config)
    elif modeltype == 'transformer-base-use-seq':
        mdl = xformer_base_use_seq(config)
    elif modeltype == 'transformer-base-simile':
        mdl = xformer_base_simile(config)
    elif modeltype == 'setransformer':
        mdl = sexformer(config)
    elif modeltype == 'setransformer-use-seq':
        mdl = sexformer_use_seq6(config)
    elif modeltype == 'setransformer-simile':
        mdl = sexformer_simile(config)
    elif modeltype == 'setransformer-bleu-base':
        mdl = sexformer_bleu_base(config)
    else:
        print("{} is not a valid model type".format(modeltype))
        exit(1)
        
    return mdl.create_model()
