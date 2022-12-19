import json

def record_list(fn, list):
    print('writing to \"' + fn + '\"')

    with open(fn, 'w') as f:
        f.write('\n'.join(list))

def record_dict(fn, dict):
    print('writing to \"' + fn + '\"')

    with open(fn, 'w') as f:
        for key in dict.keys():
            s = str(key) + ': ' + str(dict[key]) + '\n'
            f.write(s)

def recordInfo(proj_list, version):

    # label names in given dataset
    prefix = 'prefix'
    postfix = 'postfix'
    label_type = 'label-type'
    label_len = 'label-len'
    
    # record information about given dataset
    project_data = []
    token_choices = []
    sc_tokens = []
    max_len = -1
    label_len_dict = {}

    for fn in proj_list:
        print('checking project \"' + fn + '\"')

        with open('../data/'+ version + '/' + fn, 'r') as f:
            lines = f.readlines()

        for line in lines:
            json_data = json.loads(line.rstrip().replace("\'", "\""))

            if int(json_data[label_type][0]) not in token_choices:
                print('added choice \"' + str(json_data[label_type][0]) + '\"')
                token_choices.append(int(json_data[label_type][0]))

            for tok in json_data[prefix]:
                if int(tok) not in sc_tokens:
                    print('added source code prefix token to list of tokens\"' + str(tok) + '\"')
                    sc_tokens.append(int(tok))
            
            for tok in json_data[postfix]:
                if int(tok) not in sc_tokens:
                    print('added source code postfix token to list of tokens\"' + str(tok) + '\"')
                    sc_tokens.append(int(tok))
            
            if len(json_data[prefix]) > max_len: max_len = len(json_data[prefix])
            if len(json_data[postfix]) > max_len: max_len = len(json_data[postfix])

            if json_data[label_len] not in label_len_dict:
                label_len_dict[json_data[label_len]] = 1
            else:
                label_len_dict[json_data[label_len]] += 1
        
        project_data.append(fn + ': \t\t' + str(len(lines)))

    # token_choices.append('213')
    # sc_tokens.append('213')

    token_choices.sort()
    sc_tokens.sort()
    token_choices = list(map(str, token_choices))
    sc_tokens = list(map(str, sc_tokens))
    
    record_list('../record/project_data', project_data)
    record_list('../record/token_choices', token_choices)
    record_list('../record/source_code_tokens', sc_tokens)
    record_list('../record/max_len', [str(max_len)])
    record_dict('../record/label_len', label_len_dict)