import time

#out_dir = 'out-owt-gpt2mini'
out_dir = 'out-funcom-gpt2-e1'
eval_interval = 100
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'funcom'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'funcom'
init_from = 'gpt2'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

#n_layer = 6
#n_head = 6
#n_embd = 384
#dropout = 0.2

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 4 #16
gradient_accumulation_steps = 32
max_iters = 5600 # 172394 training samples

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
