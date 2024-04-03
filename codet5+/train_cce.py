import os
import pprint
import argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
import torch.nn as nn
import torch


class CustomCCETrainer(Trainer):
    def __init__(self,  **kwargs,):
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

def run_training(args, model, train_data, val_data):
    print(f"Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        save_strategy='epoch',

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_steps=args.lr_warmup_steps,

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_total_limit=1,

        dataloader_drop_last=True,
        dataloader_num_workers=4,

        eval_steps=args.eval_step,
        deepspeed=args.deepspeed,
        fp16=args.fp16,
    )

    trainer = CustomCCETrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
    )

    trainer.train()

    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')


def load_tokenize_data(args):
    # Load and tokenize data
        # Example code to load and process code_x_glue_ct_code_to_text python dataset for code summarization task
        #datasets = load_dataset("code_x_glue_ct_code_to_text", 'python', split="train")
    data_files = {"train": args.train_dataset, "val":args.val_dataset}
    datasets = load_dataset("json", data_files=data_files)
    tokenizer = AutoTokenizer.from_pretrained(args.load)

    def preprocess_function(examples):
        source = [ex for ex in examples["code"]]
        target = [ex for ex in examples["summary"]]
        model_inputs = tokenizer(source, max_length=args.max_source_len, padding="max_length", truncation=True)
        labels = tokenizer(target, max_length=args.max_target_len, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"].copy()
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
        ]
        return model_inputs

    train_val_data = datasets.map(
        preprocess_function,
        batched=True,
        #remove_columns=datasets.column_names,
        num_proc=64,
        load_from_cache_file=False,
    )
    print(f'  ==> Loaded {len(train_val_data)} samples')
    #train_data.save_to_disk(args.cache_data)
    #print(f'  ==> Saved to {args.cache_data}')
    return train_val_data


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # Save command to file
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    # Load and tokenize data using the tokenizer from `args.load`. If the data is already cached, load it from there.
    # You can customize this function to load your own data for any Seq2Seq LM tasks.
    train_val_data = load_tokenize_data(args)
    train_data = train_val_data["train"]
    val_data = train_val_data["val"]

    if args.data_num != -1:
        train_data = train_data.select([i for i in range(args.data_num)])

    # Load model from `args.load`
    model = AutoModelForSeq2SeqLM.from_pretrained(args.load)
    print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    run_training(args, model, train_data, val_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on Seq2Seq LM task")
    
    parser.add_argument('--train-dataset', default="dataset/funcom_q90_train.json", action='store_true')
    parser.add_argument('--val-dataset', default="dataset/funcom_q90_val.json", action='store_true')


    parser.add_argument('--data-num', default=-1, type=int)
    parser.add_argument('--max-source-len', default=1024, type=int)
    parser.add_argument('--max-target-len', default=50, type=int)
    #parser.add_argument('--cache-data', default='cache_data/summarize_python', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str)

    # Training
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr-warmup-steps', default=200, type=int)
    parser.add_argument('--batch-size-per-replica', default=8, type=int)
    parser.add_argument('--grad-acc-steps', default=4, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')

    # Logging and stuff
    parser.add_argument('--save-dir', default="saved_models/summarize_java", type=str)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('--save-freq', default=500, type=int)
    

    parser.add_argument('--eval-step', default=500, type=int)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
