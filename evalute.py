from datasets import load_dataset
import openai
import os
import pandas as pd
import json
import numpy as np
import time
import sys

print('Argument List:' + str(sys.argv))
i = int(sys.argv[1])

# dev_data = []
# with open("WiC_dataset/dev/dev.data.txt") as file:
#     lines = file.readlines()
#     for line in lines:
#         l = line.strip().split("\t")
#         key, pos, locs, s1, s2 = l[0], l[1], l[2], l[3], l[4]
#         dev_data.append((key, s1, s2))

# dev_gold = []
# with open("WiC_dataset/dev/dev.gold.txt") as file:
#     lines = file.readlines()
#     dev_gold = ["True" if l.strip("\n") == "T" else "False" for l in lines]

dev_data = []
with open("WiC_dataset/test/test.data.txt") as file:
    lines = file.readlines()
    for line in lines:
        l = line.strip().split("\t")
        key, pos, locs, s1, s2 = l[0], l[1], l[2], l[3], l[4]
        dev_data.append((key, s1, s2))

dev_gold = []
with open("WiC_dataset/test/test.gold.txt") as file:
    lines = file.readlines()
    dev_gold = ["True" if l.strip("\n") == "T" else "False" for l in lines]


dev = list(zip(dev_data, dev_gold))

train_data = []
with open("WiC_dataset/train/train.data.txt") as file:
    lines = file.readlines()
    for line in lines:
        l = line.strip().split("\t")
        key, pos, locs, s1, s2 = l[0], l[1], l[2], l[3], l[4]
        train_data.append((key, s1, s2))

train_gold = []
with open("WiC_dataset/train/train.gold.txt") as file:
    lines = file.readlines()
    train_gold = ["True" if l.strip("\n") == "T" else "False" for l in lines]

train = list(zip(train_data, train_gold))


# There should be a file called 'tokens.json' inside the same folder as this file
token_path = 'tokens.json'
if not os.path.isfile(token_path):
    raise Exception(f"{token_path} not found!")
with open(token_path) as f:
    # If you get an error here, it means your token is formatted incorrectly. Did you put it in quotes?
    tokens = json.load(f)
    openai.organization = tokens['openai_organization']
    openai.api_key = tokens['openai_api_key']


def formation(data_point):
    keyword, s1, s2 = data_point
    nl = '\n'
    return f'Keyword: {keyword}{nl}Sentence1: {s1}{nl}Sentence2: {s2}' 

def classify(examples_data, examples_label, querry_point):

    gpt_message = [
    {"role": "system", "content": "You are a word-sense disambiguation system. Given a key word and two sentences using that word, answer True if the key word has \
        the same word sense in both sentences and False otherwise. The key may appear in slightly different forms in the two sentences."},
    {"role": "user", "content": formation(examples_data[0])},
    {"role": "assistant", "content": examples_label[0]},
    {"role": "user", "content": formation(examples_data[1])},
    {"role": "assistant", "content": examples_label[1]},
    {"role": "user", "content": formation(examples_data[2])},
    {"role": "assistant", "content": examples_label[2]},
    {"role": "user", "content": formation(examples_data[3])},
    {"role": "assistant", "content": examples_label[3]},
    {"role": "user", "content": formation(examples_data[4])},
    {"role": "assistant", "content": examples_label[4]},
    {"role": "user", "content": formation(querry_point)}
    ]

    # print(gpt_message)

    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=gpt_message
    )

    output = response['choices'][0]['message']['content']
    return output

NUM_EXAMPLES = 5
indices = np.random.choice(np.arange(len(train)), NUM_EXAMPLES, replace=False)
example_data_test = list(np.array(train_data)[indices])
examples_label_test = list(np.array(train_gold)[indices])

print(classify(example_data_test, examples_label_test, dev_data[0]))
print(f'real answer: {dev_gold[0]}')
print(example_data_test, examples_label_test)



TP, FP, FN, TN = 0, 0, 0, 0
TP_data, FP_data, FN_data, TN_data = [], [], [], []


while i < len(dev_data):

    dev_data_single = dev_data[i]
    dev_data_gold = dev_gold[i]
    print(f'round: {i}, dev_data: {dev_data_single}, dev_label: {dev_data_gold}')

    indices = np.random.choice(np.arange(len(train)), NUM_EXAMPLES, replace=False)
    example_data_test = list(np.array(train_data)[indices])
    examples_label_test = list(np.array(train_gold)[indices])

    while all([single_lable == "True" for single_lable in examples_label_test]) or all([single_lable == "False" for single_lable in examples_label_test]):
        indices = np.random.choice(np.arange(len(train)), NUM_EXAMPLES, replace=False)
        example_data_test = list(np.array(train_data)[indices])
        examples_label_test = list(np.array(train_gold)[indices])
    
    print(example_data_test, examples_label_test)
    try:
        result = classify(example_data_test, examples_label_test, dev_data_single)
        print(f'prediction: {result}')
    
        if dev_data_gold == "True":
            if result == "True":
                TP += 1
            else:
                FN += 1
        else:
            if result == "True":
                FP += 1
            else:
                TN += 1
        print(TP, FP, FN, TN)
        i += 1
    except:
        print("error")
        time.sleep(3)
        continue

print("success")