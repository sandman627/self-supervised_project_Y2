sentence = f'''
[D]: dddd
[E]
[C]: cccc
[Q]: qqqq
[A]: aaaa

[P]
[E]
[C]: ccc1
[Q]: qqq1
[A]: aaa1

[P]
[C]: ccc2
[Q]: qqq2
[A]: aaa2

[P]
[C]: ccc3
[Q]: qqq3
[A]: aaa3

[P]
[C]: ccc4
[Q]: qqq4
[A]: aaa4

[P]
[C]: ccc5
[Q]: qqq5
[A]: aaa5
'''
problem_word = "[P]"
answer_word = "[A]:"

# Split the sentence based on the search word
problem_parts = sentence.split(problem_word)[1:]
print(f"Only Problems : {problem_parts}")


answers = []
if len(problem_parts) > 1:
    for num, prob_int in enumerate(problem_parts):
        print(f"Prob Int {num} : {prob_int}")
        pb = prob_int.strip().split(answer_word, 1)
        answer = pb[1].strip()
        print(f"Answer {num} : {answer}")
        answers.append(answer)
    print(f"Answer : {answers[0]}")
else:
    print("No Answer")




ddd = 'q\n[]q'

print("ddddddddd : ",ddd.split('\n')[0])