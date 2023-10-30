import os
import pickle
from tqdm import tqdm
from typing import List
import string, re


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
qa_srl_dataset = load_dataset("qa_srl")
# exit()

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


def pipe(question, context):
    # print("Context : ", context)
    # print("Question : ", question)
    
    question_as_sentence = ' '.join(question).replace("_", "")
    # print("Question as Sentence : ", question_as_sentence)
    
    prompt_template=f'''
    [Description]: Find the answer from the Context as phrase. Never explain reasoning. Give only simple short phrase. Do Not Repeat the Example. Solve the problem just like the given example. What will be the answer?
    [Exmaple]
    [Context]: The proposal intends to use wind-driven pumps to inject oxygen into waters at , or around , 130m below sea level .
    [Question]: what intends to do something ?
    [Answer]: The proposal
    
    [Problem]
    [Context]: {context}
    [Question]: {question_as_sentence}
    [Answer]: '''
    
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output_ids = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    output = tokenizer.decode(output_ids[0])
    print(f"output : \n{output}\n")
    
    output = get_answer_only(output)
    print(f"Model Prediction : {output}")
    
    return output


def get_answer_only(sentence, problem_indicating_word:str='[Problem]', answer_indicating_word:str='[Answer]:'):
    problem_parts = sentence.split(problem_indicating_word)

    answers=[]
    if len(problem_parts) > 1:
        for num, prob_set in enumerate(problem_parts[1:]):
            # print(f"Prob Int {num} : {prob_set}")
            single_pb = prob_set.split(answer_indicating_word, 1)
            if len(single_pb) > 1:
                answer = single_pb[1].strip()
                # print(f"Answer {num} : {answer}")
                answer = answer.split('[')[0]  # just in case of [Description] or [Example] shows right after the answer.
                answer = answer.split('\n')[0]  # just in case of [Description] or [Example] shows right after the answer.
                answers.append(answer)
            else:
                # print(f"Answer {num} : No Answer, Cutted")
                pass
        return answers[0]
    else:
        print("No Answer")
        return "No Answer"


# these functions are heavily influenced by the HF squad_metrics.py script
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)


def get_F1_score(prediction_list:List[str], labels_list:List[List[str]]):
    # prediction_list is list[str], but label_list is list[list[str]]
    f1_scores = []
    for prediction, labels in zip(prediction_list, labels_list):
        max_f1_score = 0
        for label in labels:
            f1_score = compute_f1(prediction, label)              
            if f1_score >= max_f1_score:
                max_f1_score = f1_score
        f1_scores.append(max_f1_score)
    
    return sum(f1_scores)/len(f1_scores)



# Run Evaluation
episode_num = 0
dataset_answers = []
dataset_references = []
model_answers = []
model_predictions = []
for data in tqdm(qa_srl_dataset['test']):
    print("data : \n", data)
    answers = data['answers']
    predicate = data['predicate']
    predicate_idx = data['predicate_idx']
    question = data['question']
    sent_id = data['sent_id']
    sentence = data['sentence']

    # Get dataset label
    dataset_answers.append(answers)

    # Get prediction
    model_prediction = pipe(question=question, context=sentence)
    model_answers.append(model_prediction)

    # Format for SQuAD metric
    dataset_reference = {'answers':{'answer_start': [predicate_idx], 'text':[answers]}, 'id':sent_id}
    dataset_references.append(dataset_reference)    
    model_prediction = {'prediction_text':model_prediction, 'id':sent_id}
    model_predictions.append(model_prediction)

    # n-times inference
    episode_num += 1
    if episode_num > 3:
        break

print("Model Predictions : ", model_answers)
print("Dataset Labels : ", dataset_answers)

f1_score = get_F1_score(model_answers, dataset_answers)
print(f"F1 Score : {f1_score}")


exit()
with open('results/Semantic_Role_Labeling/model_predictions.pkl', 'wb') as pf:
    pickle.dump(exp_results, pf, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    print("Testing : ", os.path.basename(__file__)) 