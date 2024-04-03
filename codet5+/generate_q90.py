from transformers import AutoTokenizer, T5ForConditionalGeneration, GenerationConfig
import pickle
import json
from tqdm import tqdm
import argparse

device = "cuda"






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--test-file-name', default='./dataset/funcom_q90_test.json', type=str)
    args = parser.parse_args()

    model_name = args.model_name
    test_file_name = args.test_file_name

    prediction_filename = f"{model_name}.txt"

    funcom_q90_test_file = open(test_file_name)
    funcom_q90_test = json.load(funcom_q90_test_file)

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m")
    model = T5ForConditionalGeneration.from_pretrained(f"saved_models/{model_name}/final_checkpoint").to(device)
    pf = open(f'predictions/{prediction_filename}', 'w')




for f in tqdm(funcom_q90_test[:]):
    fid = f['fid']
    code = f['code']
    
    code = tokenizer(code, max_length = 1024, return_tensors="pt",truncation=True).input_ids.to(device)

    output = model.generate(code, max_length=50)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    pf.write(f'{fid}\t{output}\n')


pf.close()


