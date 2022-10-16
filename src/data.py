import json
import numpy as np

def getTrainData(proj_list, target_project):

    total_file = 'total'

    prefix = []
    postfix = []
    label_type = []
    label_len = []

    for proj in proj_list:
        
        if proj == target_project or proj == total_file: continue

        print('Getting data for \"' + target_project + '\" from \"' + proj + '\"')

        with open('../data/' + proj, 'r') as f:

            lines = f.readlines()
        
        for line in lines:

            json_data = json.loads(line.rstrip())

            prefix.append(json_data['prefix'])
            postfix.append(json_data['postfix'])
            label_type.append(json_data['label-type'])
            label_len.append(json_data['label-len'])
    
    return np.array(prefix), np.array(postfix), np.array(label_type), np.array(label_len)

def getTestData(target_project):
    prefix = []
    postfix = []
    label_type = []
    label_len = []

    with open('../data/' + target_project, 'r') as f:

        lines = f.readlines()
    
    for line in lines:
        json_data = json.loads(line.rstrip())

        prefix.append(json_data['prefix'])
        postfix.append(json_data['postfix'])
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