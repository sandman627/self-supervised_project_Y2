import os
import pickle
from tqdm import tqdm

from datasets import load_dataset
import evaluate
from evaluate import evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

'''
Summarization: 
    dataset: CNN/DM
    metric: average three ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
'''

# Dataset
dataset = load_dataset("cnn_dailymail", '3.0.0')

# Metric
rouge_metric = evaluate.load('rouge')

# Model
model_name_or_path = "TheBloke/Llama-2-70B-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

def summerizer(input_text):
    prompt = "Summarize the Following content in less than 120 words."
    prompt_template=f"{prompt}{input_text}"
    
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    
    print(tokenizer.decode(output[0]))
    return tokenizer.decode(output[0])



# Run Evaluation
max_words = 0
max_length = 0

dataset_summaries = []
model_summaries = []

test_case_num = 0
skipped_case_nums = []
skipped_num = 0
for data in tqdm(dataset['test']):
    article = data['article']
    data_hightlight = data['highlights']

    n_words = len(article.split())
    n_length = len(article)

    if n_words > max_words:
        max_words = n_words
    
    if n_length > max_length:
        max_length = n_length

    try:
        summary = summerizer(article)
        dataset_summaries.append(data_hightlight)
        model_summaries.append(summary)
    except:
        skipped_case_nums.append(test_case_num)        
        skipped_num += 1

    test_case_num += 1
    # if test_case_num == 100:
    #     break
    break

print("max words : ", max_words)
print("max length : ", max_length)

rouge_scores = rouge_metric.compute(predictions=model_summaries, references=dataset_summaries)
for key, value in rouge_scores.items():
    print(f"{key} : {value}")

print("Skipped Cases : ", skipped_case_nums)
print("num of Skipped Cases : ", skipped_num)


if __name__ == "__main__":
    print("Testing : ", os.path.basename(__file__)) 