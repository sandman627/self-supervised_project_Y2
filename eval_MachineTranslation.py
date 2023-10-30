import os

from tqdm import tqdm

from datasets import load_dataset
import evaluate
from evaluate import evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

'''
Machine Translation: 
    dataset: IWSLT (2016, German to English)
    metric: BLEU
'''

# Dataset
dataset = load_dataset()

# Metric
bleu_metric = evaluate.load('bleu')

# Model
model_name_or_path = "TheBloke/Llama-2-70B-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

def pipe(question, context):
    prompt_template=f'''
    Context: {context}\n
    Question: Fill in the blank of given sentence. {question}\n
    Answer: 
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

episode_num = 0
predictions = []
references = []
for data in tqdm(dataset['test']):
    
    references.append(data['label'])
    
    predictions.append()
    
bleu_score = bleu_metric.compute(predictions=predictions, references=references)
print(bleu_score)





if __name__ == "__main__":
    print("Testing : ", os.path.basename(__file__)) 