from datasets import load_dataset

import evaluate
from evaluate import evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline



'''
Machine Translation: 
    dataset: IWSLT (2016, German to English)
    metric: BLEU
        
Summarization: 
    dataset: CNN/DM
    metric: average three ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)

Semantic Role Labeling: 
    dataset: QA-SRL
    metric: nF1
'''

metrics_list = evaluate.list_evaluation_modules()
len(metrics_list)
print("Metric List: ", len(metrics_list))
print(metrics_list)



model_name_or_path = "TheBloke/Llama-2-70B-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = "Tell me about AI"
prompt_template=f'''{prompt}

'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

print(pipe(prompt_template)[0]['generated_text'])
