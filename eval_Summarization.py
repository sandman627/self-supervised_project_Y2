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
    # print(f"input text : \n{input_text}\n\n\n\n")
    prompt = "Please summarize the previous content in one or two sentences, less than 300 words."
    prompt_template=f'''
    Content: {input_text}
    Question: {prompt}
    Summary: 
    '''
    prompt_len = len(prompt_template)
    
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output_ids = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    # print(f"output ids: {type(output_ids)}\n{output_ids}\n\n\n\n")
    output = tokenizer.decode(output_ids[0])
    output_without_prompt = output[prompt_len:]
    # print(f"output : \n{output}\n\n\n\n")
    # print(f"output summary : \n{output_only_summary}\n\n\n\n")
    return output_without_prompt



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

    # try:
    summary = summerizer(article)
    dataset_summaries.append(data_hightlight)
    model_summaries.append(summary)
    print("summary len : ", len(summary))
    print("reference len : ", len(data_hightlight))        
    # except:
    #     skipped_case_nums.append(test_case_num)        
    #     skipped_num += 1

    test_case_num += 1
    # if test_case_num == 100:
    #     break
    # break

print("Num of summary : ", len(model_summaries))
print("Skipped Cases : ", skipped_case_nums)
print("num of Skipped Cases : ", skipped_num)
print("max words : ", max_words)
print("max length : ", max_length)

rouge_scores = rouge_metric.compute(predictions=model_summaries, references=dataset_summaries)
for key, value in rouge_scores.items():
    print(f"{key} : {value}")
rouge_scores_average = (rouge_scores['rouge1'] + rouge_scores['rouge2'] + rouge_scores['rougeL'])/3

print("ROUGE Score: ", rouge_scores_average)

if __name__ == "__main__":
    print("Testing : ", os.path.basename(__file__)) 