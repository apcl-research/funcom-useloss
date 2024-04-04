from transformers import Trainer
import numpy as np
import torch.nn as nn
import torch
import os
from huggingface_hub import snapshot_download
import tensorflow_hub as hub
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

class CustomCCETrainer(Trainer):
    def __init__(self, o_data, customtokenizer, beta, **kwargs,):
        super().__init__(**kwargs)
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        input_ids = inputs.get('input_ids')
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()
        
        
        
        input_mask = torch.where(shift_labels.view(-1)!=-100, 1, 0)
        
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        loss = loss * input_mask
        
        
        n_of_none_zero = torch.count_nonzero(loss)
        loss = loss / n_of_none_zero
        loss = torch.sum(loss)



        return (loss, outputs) if return_outputs else loss

class CustomUSESeqTrainer1(Trainer):
    def __init__(self, o_data, customtokenizer, beta, **kwargs,):
        super().__init__(**kwargs)
        self.o_data = o_data
        self.tokenizer = customtokenizer
        use_model_path = snapshot_download(repo_id="Dimitre/universal-sentence-encoder")
        self.use_model =  hub.KerasLayer(handle=use_model_path)
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        input_ids = inputs.get('input_ids')
        outputs = model(**inputs)
        logits = outputs.get("logits")
        #prompt_len_list = seld.o_data['user_prompt_len']
        #index_list = []
        #for i in range(labels.shape[0]):
        #    for j in range(len(self.o_data['input_ids'])):
        #        if(self.o_data['input_ids'][j] == input_ids[i].cpu().numpy().tolist() ):#.nonzero(as_tuple=True)[0]
        #            index_list.append(j)
        #            break

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()
        arg_shift_logits = torch.argmax(shift_logits, -1)
        
        sentence = torch.argmax(shift_logits, dim=-1)
        true_sentence = shift_labels.clone()
        true_sentence[true_sentence==-100]=0

        cos = torch.nn.CosineSimilarity(dim=-1)
        
        pred_sentence = self.tokenizer.batch_decode(sentence)
        true_sentence = self.tokenizer.batch_decode(true_sentence)

        for i in range(len(true_sentence)):
            true_sentence[i] = true_sentence[i].split('<unk>')
            for j in range(len(true_sentence[i])- 1, 0, -1 ):
                temp_sentence = true_sentence[i][j].strip() 
                temp_sentence = temp_sentence.split(' ')
                if(temp_sentence[0] == '<s>'):
                    temp_sentence = ' '.join(temp_sentence)
                    true_sentence[i] = temp_sentence
                    pred_sentence[i] = pred_sentence[i][:len(true_sentence[i])].strip()
                    break
                else:
                    continue
        pred_emb = self.use_model(pred_sentence).numpy()
        true_emb = self.use_model(true_sentence).numpy()
        pred_emb = torch.tensor(pred_emb)
        true_emb = torch.tensor(true_emb)

        use_score = cos(pred_emb, true_emb)
        use_score = use_score.to('cuda')
        use_score = torch.unsqueeze(use_score, dim=1)
        use_score = torch.repeat_interleave(use_score, shift_labels.shape[1], dim=1)
        weight_mask = torch.where(use_score > 0, torch.where(shift_labels == arg_shift_logits, 1.0, 0.), torch.where(shift_labels == arg_shift_logits, 0., 1.0))
        use_score = use_score * weight_mask
        weight = torch.exp(use_score/self.beta)
        weight = weight.view(-1)
        
        input_mask = torch.where(shift_labels.view(-1)!=-100, 1, 0)
        
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss * weight

        loss = loss * input_mask

        
        n_of_none_zero = torch.count_nonzero(loss)
        loss = loss / n_of_none_zero
        loss = torch.sum(loss)



        return (loss, outputs) if return_outputs else loss


class CustomUSESeqTrainer(Trainer):
    def __init__(self, customtokenizer, beta, **kwargs,):
        super().__init__(**kwargs)
        self.tokenizer = customtokenizer
        use_model_path = snapshot_download(repo_id="Dimitre/universal-sentence-encoder")
        self.use_model =  hub.KerasLayer(handle=use_model_path)
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()
        arg_shift_logits = torch.argmax(shift_logits, -1)
        
        sentence = torch.argmax(shift_logits, dim=-1)
        true_sentence = shift_labels.clone()
        true_sentence[true_sentence==-100]=0

        cos = torch.nn.CosineSimilarity(dim=-1)
        
        pred_sentence = self.tokenizer.batch_decode(sentence)
        true_sentence = self.tokenizer.batch_decode(true_sentence)

        for i in range(len(true_sentence)):
            true_sentence[i] = true_sentence[i].split('<unk>')[0].strip()
            pred_sentence[i] = pred_sentence[i][:len(true_sentence[i])].strip()
        pred_emb = self.use_model(pred_sentence).numpy()
        true_emb = self.use_model(true_sentence).numpy()
        pred_emb = torch.tensor(pred_emb)
        true_emb = torch.tensor(true_emb)

        use_score = cos(pred_emb, true_emb)
        use_score = use_score.to('cuda')
        use_score = torch.unsqueeze(use_score, dim=1)
        use_score = torch.repeat_interleave(use_score, shift_labels.shape[1], dim=1)
        weight_mask = torch.where(use_score > 0, torch.where(shift_labels == arg_shift_logits, 1.0, 0.), torch.where(shift_labels == arg_shift_logits, 0., 1.0))
        use_score = use_score * weight_mask
        weight = torch.exp(use_score/self.beta)
        weight = weight.view(-1)
         

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss * weight
        
        n_of_none_zero = torch.count_nonzero(loss)
        loss = loss / n_of_none_zero
        loss = torch.sum(loss)



        return (loss, outputs) if return_outputs else loss
