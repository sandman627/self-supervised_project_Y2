import os
import pickle
from tqdm import tqdm

from datasets import load_dataset
import evaluate
from evaluate import evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

'''
Semantic Role Labeling: 
    dataset: QA-SRL
    metric: nF1 (normalized F1)
'''

# Dataset
with open("/home/sandman/self-supervised_project/data/qa_srl_test.pkl", 'rb') as f:
    qa_srl_dataset = pickle.load(f)

# Metric
squad_metric = evaluate.load('squad')
f1_metric = evaluate.load('f1')
em_metric = evaluate.load('exact_match')

# Model
model_name_or_path = "TheBloke/Llama-2-70B-GPTQ"
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

print("*** Pipeline:")
pipe = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)


# Run Evaluation
episode_num = 0
dataset_answers = []
model_answers = []
dataset_references = []
model_predictions = []
for data in tqdm(qa_srl_dataset):
    answers = data['answers']
    predicate = data['predicate']
    predicate_idx = data['predicate_idx']
    question = data['question']
    sent_id = data['sent_id']
    sentence = data['sentence']

    question_as_sentence = ' '.join(question)

    dataset_answers.append(answers)
    reference = {'answers':{'answer_start': [predicate_idx], 'text':[answers]}, 'id':sent_id}
    dataset_references.append(reference)

    result = pipe(question=question_as_sentence, context=sentence)

    model_answers.append(result['answer'])
    prediction = {'prediction_text':result['answer'], 'id':sent_id}
    model_predictions.append(prediction)

    # print(question_as_sentence)
    # print(sentence)
    # print(answers)
    # print(result['answer'])
    # exit()

    # episode_num += 1
    # if episode_num > 10:
    #     break

# print(em_metric.compute(predictions=model_answers, references=dataset_answers))
print(squad_metric.compute(predictions=model_predictions, references=dataset_references))


if __name__ == "__main__":
    print("Testing : ", os.path.basename(__file__)) 