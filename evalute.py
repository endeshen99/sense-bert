from datasets import load_dataset
import openai
import os
import pandas as pd

dev_data = []
with open("WiC_dataset/dev/dev.data.txt") as file:
    lines = file.readlines()
    for line in lines:
        l = line.strip().split("\t")
        key, pos, locs, s1, s2 = l[0], l[1], l[2], l[3], l[4]
        dev_data.append((key, s1, s2))

dev_gold = []
with open("WiC_dataset/dev/dev.data.txt") as file:
    lines = file.readlines()
    dev_gold = ["True" if l.strip("\n") == "T" else "False" for l in lines]

train_data = []
with open("WiC_dataset/dev/dev.data.txt") as file:
    lines = file.readlines()
    for line in lines:
        l = line.strip().split("\t")
        key, pos, locs, s1, s2 = l[0], l[1], l[2], l[3], l[4]
        train_data.append((key, s1, s2))

train_gold = []
with open("WiC_dataset/dev/dev.data.txt") as file:
    lines = file.readlines()
    train_gold = ["True" if l.strip("\n") == "T" else "False" for l in lines]


# There should be a file called 'tokens.json' inside the same folder as this file
token_path = 'tokens.json'
if not os.path.isfile(token_path):
    raise Exception(f"{token_path} not found!")
with open(token_path) as f:
    # If you get an error here, it means your token is formatted incorrectly. Did you put it in quotes?
    tokens = json.load(f)
    openai.organization = tokens['openai_organization']
    openai.api_key = tokens['openai_api_key']


def classify(message, terrorism_examples, safe_examples):

    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
    {"role": "system", "content": "You are a content moderation system. Classify each input as either terrorism or safe."},
    {"role": "user", "content": terrorism_examples[0]},
    {"role": "assistant", "content": "terrorism"},
    {"role": "user", "content": safe_examples[0]},
    {"role": "assistant", "content": "safe"},
    {"role": "user", "content": terrorism_examples[1]},
    {"role": "assistant", "content": "terrorism"},
    {"role": "user", "content": safe_examples[1]},
    {"role": "assistant", "content": "safe"},
    {"role": "user", "content": terrorism_examples[2]},
    {"role": "assistant", "content": "terrorism"},
    {"role": "user", "content": safe_examples[2]},
    {"role": "assistant", "content": "safe"},
    {"role": "user", "content": message}
    ]
    )

    output = response['choices'][0]['message']['content']
    return output


def evaluate(terrorism_data, safe_data, percentage_test = 0.001, num_example = 6):
    terrorism_cutoff = int(np.ceil(len(terrorism_data) * percentage_test))
    terrorism_train, terrorism_test = terrorism_data[terrorism_cutoff:], terrorism_data[:terrorism_cutoff]
    safe_cutoff = int(np.ceil(len(safe_data) * percentage_test))
    safe_train, safe_test = safe_data[safe_cutoff:], safe_data[:safe_cutoff]


    TP, FP, FN, TN = 0, 0, 0, 0
    TP_tweet, FP_tweet, FN_tweet, TN_tweet = [], [], [], []
    print("evaluating terrorism")
    for terrorism_tweet in terrorism_test:
        result = classify(terrorism_tweet,
                          np.random.choice(terrorism_train, size=3, replace=False),
                          np.random.choice(safe_train, size=3, replace=False))
        if result == "terrorism":
            TP += 1
            TP_tweet.append([terrorism_tweet])
        elif result == "safe":
            FN += 1
            FN_tweet.append([terrorism_tweet])
            print(terrorism_tweet)
        else:
            print(result)

    print("evaluating safe tweets")
    print(len(safe_test))
    for safe_tweet in safe_test:
        print(safe_tweet)
        result = classify(safe_tweet,
                          np.random.choice(terrorism_train, size=3, replace=False),
                          np.random.choice(safe_train, size=3, replace=False))
        if result == "terrorism":
            FP += 1
            FP_tweet.append([safe_tweet])
            print(safe_tweet)
        elif result == "safe":
            TN += 1
            TN_tweet.append([safe_tweet])
        else:
            print(result)

    print(TP, FP, FN, TN)

    filename = "tweets_classified.csv"

    with open(filename, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(TP_tweet)
        csvwriter.writerows([[f'TP: {TP}']])
        csvwriter.writerows(FN_tweet)
        csvwriter.writerows([[f'FN: {FN}']])
        csvwriter.writerows(FP_tweet)
        csvwriter.writerows([[f'FP: {FP}']])
        csvwriter.writerows(TN_tweet)
        csvwriter.writerows([[f'TN: {TN}']])

evaluate(terrorism_data, safe_data, percentage_test=0.01)