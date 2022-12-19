import json
import numpy as np
from collections import deque
import csv

def insertSOSandEOS(tok_list):
    tmp = deque(tok_list)
    tmp.appendleft(213)
    tmp.append(214)
    tmp = list(tmp)
    return tmp

def insertEOS(tok_list, idx):
    tmp = tok_list
    tmp.insert(idx, 1)
    return tmp

def getTrainData(proj_list, target_project, version):

    total_file = 'total'

    prefix = []
    postfix = []
    label_type = []
    label_len = []

    for proj in proj_list:
        
        # if proj == target_project or proj == total_file: continue

        # don't remove target file for training web model
        if proj == total_file: continue

        print('Getting data for \"' + target_project + '\" from \"' + proj + '\"')

        with open('../data/' + version + '/' + proj, 'r') as f:
            lines = f.readlines()
        
        for line in lines:

            json_data = json.loads(line.rstrip())

            # prefix.append(insertSOSandEOS(json_data['prefix']))
            # postfix.append(insertSOSandEOS(json_data['postfix']))

            prefix.append(json_data['prefix'])

            # remove first token of postfix and insert paddin at the end
            postfix_data = json_data['postfix']
            postfix_data.pop()
            postfix_data.insert(0, 0)
            postfix.append(postfix_data)

            label_type.append(json_data['label-type'])
            label_len.append(json_data['label-len'])
    
        # ------------------------------------------------------
        # break for reducing test time for quick development
        break
    
    return np.array(prefix), np.array(postfix), np.array(label_type), np.array(label_len)

def getTestData(target_project, version):
    prefix = []
    postfix = []
    label_type = []
    label_len = []

    with open('../data/' + version + '/' + target_project, 'r') as f:

        lines = f.readlines()
    
    for line in lines:

        json_data = json.loads(line.rstrip())

        # prefix.append(insertSOSandEOS(json_data['prefix']))
        # postfix.append(insertSOSandEOS(json_data['postfix']))

        prefix.append(json_data['prefix'])

        # remove first token of postfix and insert paddin at the end
        postfix_data = json_data['postfix']
        postfix_data.pop()
        postfix_data.insert(0, 0)
        postfix.append(postfix_data)

        label_type.append(json_data['label-type'])
        label_len.append(json_data['label-len'])
    
    return np.array(prefix), np.array(postfix), np.array(label_type), np.array(label_len)

def getInfo():

    max_len = 0
    source_code_tokens = []
    token_choices = []

    with open('../record/max_len', 'r') as f:
        max_len = int(f.readline().rstrip())
    
    with open('../record/source_code_tokens', 'r') as f:
        source_code_tokens = [int(line.rstrip()) for line in f]
    
    with open('../record/token_choices', 'r') as f:
        token_choices = [int(line.rstrip()) for line in f]

    return max_len, source_code_tokens, token_choices

def getIdx2str():
    idx2str = {}

    with open('../record/token_str', 'r') as f:
        csvReader = csv.reader(f)

        for row in csvReader:
            # if row[1] != '':
            idx2str[int(row[0])] = row[1]
    
    return idx2str



def idx2str(resultsIdx):
    idx2str = getIdx2str()

    final = []

    for seq in resultsIdx:
        str_list = []

        for token in seq:
            str_list.append(idx2str[token])

        str = ' '.join(str_list) 
        final.append(str)
    
    return final