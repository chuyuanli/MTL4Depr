from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix
import ast
import numpy as np
import random
import pandas as pd

from constant import F_TOPIC, T_TRAIN, T_DEV, T_TEST, F_TEXT

MAX_LEVEL = 'block'

def parser_file_topic(topic_file):
    # read topic files and stock for each topic the corresponding line index
    # return a dictionary with the index for each topic
    topics = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]} #store nb of lines in corresponding topic index
    topic_count = [0]*10
    with open(topic_file, 'r') as instr:
        lines = instr.readlines()
        for idx, line in enumerate(lines):
            topic_nb = int(line.strip())
            topic_count[topic_nb-1] += 1 #topic number starts with 1
            topics[topic_nb].append(idx+1) #index starts with 0
    return topics

def separate_train_dev_test():
    # input is a dictionary with index of all topics
    # output is 3 lists (train, dev, test) with index
    dico_topic = parser_file_topic(F_TOPIC)

    train, dev, test = [], [], []
    train_len = 11118
    dev_len = 1000
    test_len = 1000
    total = train_len + dev_len + test_len
    train_part = train_len / total
    dev_part = dev_len / total
    test_part = test_len / total

    random.seed(2)
    
    for k, v in dico_topic.items():
        all_val = v
        test_topic_len = round(len(v) * test_part)
        dev_topic_len = round(len(v) * dev_part)
        train_topic_len = len(v) - test_topic_len - dev_topic_len
        rand_train = random.sample(all_val, k=train_topic_len)
        train.extend(rand_train) 

        devtest_val = [j for j in all_val if j not in rand_train]
        rand_dev = random.sample(devtest_val, k=dev_topic_len)
        dev.extend(rand_dev)

        test_val = [x for x in devtest_val if x not in rand_dev]
        test.extend(test_val)

    # check    
    assert len(set(dev)) == len(dev) == 1000, f"{len(train) - len(set(train))} duplicates in dev!"
    assert len(set(test)) == len(test) == 1000, f"{len(train) - len(set(train))} duplicates in dev!"
    assert len(set(train)) == len(train) == 11118, f"{len(train) - len(set(train))} duplicates in train!"
    commun = set(train).union(set(dev)).union(set(test))
    assert len(commun) != 0, f'Found duplicates in train, dev and test!'
    return train, dev, test

def original_separation():
    # in compare with the separation based on the equility of topics.
    # this separation takes the original separation from dailydialog:
    # http://yanran.li/dailydialog
    # output is the same format, train/dev/test returns a list of line indexes (starting from 1)
    train, dev, test = [], [], []
    all_text, train_text, dev_text, test_text = [], [], [], []
    with open(F_TEXT, 'r') as inst:
        all_text = inst.readlines()

    with open(T_TRAIN, 'r') as inst:
        lines = inst.readlines()
        for l in lines:
            assert l in all_text, f'{l} in train text is not found in the full text.'
            train.extend([i+1 for i, e in enumerate(all_text) if e == l]) #take all indexes
    train = list(set(train))
    # assert len(train) == 11118, f'Train text length is {len(train)}, should be 11118.'

    with open(T_DEV, 'r') as inst:
        lines = inst.readlines()
        for l in lines:
            assert l in all_text, f'{l} in validation text is not found in the full text.'
            dev.extend([i+1 for i, e in enumerate(all_text) if e == l])
    dev = list(set(dev))

    with open(T_TEST, 'r') as inst:
        lines = inst.readlines()
        for l in lines:
            assert l in all_text, f'{l} in test text is not found in the full text.'
            test.extend([i+1 for i, e in enumerate(all_text) if e == l])
    test = list(set(test))
    return train, dev, test


def test_doc_res(pred_file: str, params: dict) -> str:
    has_emo = params['has_emo']
    has_act = params['has_act']
    has_topic = params['has_topic']
    has_phq = params['has_phq']
    has_phqbi = params['has_phqbi']

    gold_e, gold_p, gold_a, gold_t = [], [], [], []
    pred_e, pred_p, pred_a, pred_t = [], [], [], []

    with open(pred_file, 'r') as instream:
        lines = instream.readlines() #each line is a dict with keys 'logits_turn','probs_turn'..."loss_turn","loss_block","loss_doc","loss"
        for l in lines:
            data = ast.literal_eval(l)
            
            if has_emo:
                gold_emo = data['label_emo']
                if not gold_emo == 'None':
                    pred_emo = data['probs_emo']
                    for i in range(len(gold_emo)):
                        for gold, pred in zip(gold_emo[i], pred_emo[i]):
                            if gold != -1:
                                gold_e.append(gold)
                                pred_e.append(pred.index(max(pred)))
            if has_phq:
                gold_phq = data['label_phq']
                if not set(gold_phq) == {-1} and not gold_phq == 'None':
                    pred_phq = data['probs_phq']
                    #fill in matrix for turn and block
                    gold_p.extend(gold_phq)
                    for pd in pred_phq:
                        pred_p.append(pd.index(max(pd)))
            if has_act:
                gold_act = data['label_act']
                if not gold_act == 'None':
                    pred_act = data['probs_act']
                    for i in range(len(gold_act)):
                        for gold, pred in zip(gold_act[i], pred_act[i]):
                            if gold != -1:
                                gold_a.append(gold)
                                pred_a.append(pred.index(max(pred)))
            if has_topic:
                gold_topic = data['label_topic']
                if not set(gold_topic) == {-1} and not gold_topic == 'None':
                    pred_topic = data['probs_topic']
                    gold_t.extend(gold_topic)
                    for pd in pred_topic:
                        pred_t.append(pd.index(max(pd)))

    if gold_e != [] and pred_e != []:
        e = confusion_matrix(y_true=gold_e, y_pred=pred_e, labels=[0, 1, 2, 3, 4, 5, 6]) #7 emotions
    if gold_a != [] and pred_a != []:
        a = confusion_matrix(y_true=gold_a, y_pred=pred_a, labels=[0, 1, 2, 3]) #4 da
    if gold_t != [] and pred_t != []:
        t = confusion_matrix(y_true=gold_t, y_pred=pred_t, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) #10 topics
    if gold_p != [] and pred_p != []:
        p = confusion_matrix(y_true=gold_p, y_pred=pred_p, labels=[0, 1]) #2 phq scores
    
    if has_emo and has_phq and has_act and has_topic:
        return (e.tolist(), a.tolist(), t.tolist(), p.tolist())
    elif has_phq and has_act:
        return (a.tolist(), p.tolist())
    elif has_phq and has_topic:
        return (t.tolist(), p.tolist())
    elif has_emo and has_phq:
        return (e.tolist(), p.tolist())
    elif has_phq:
        return p.tolist()

def _weird_division(n, d):
    return n / d if d else 0
    
def calculate_balanced_acc(confmat, params):
    # balanced acc. is the average of recall obtained on each class
    emo, act, topic, phq = None, None, None, None
    if params['has_emo'] and params['has_phq'] and params['has_act'] and params['has_topic']:
        emo = confmat[0]
        act = confmat[1]
        topic = confmat[2]
        phq = confmat[3]
    elif params['has_phq'] and params['has_act'] :
        act = confmat[0]
        phq = confmat[1]
    elif params['has_phq'] and params['has_topic']:
        topic = confmat[0]
        phq = confmat[1]
    elif params['has_emo'] and params['has_phq']:
        emo = confmat[0] #7 emo
        phq = confmat[1] #5 or 2 phq
    elif params['has_phq']:
        phq = confmat

    emo_rec_sum, phq_rec_sum, act_rec_sum, topic_rec_sum = 0.0, 0.0, 0.0, 0.0
    emo_acc, phq_acc, act_acc, topic_acc = 0.0, 0.0, 0.0, 0.0
    if phq is not None:
        for i in range(len(phq)):
            phq_rec_sum += _weird_division(phq[i][i], np.sum(phq[i]))
            phq_acc = round(phq_rec_sum / 2, 3)
    if emo is not None:
        for i in range(len(emo)):
            emo_rec_sum += _weird_division(emo[i][i], np.sum(emo[i]))
        emo_acc = round(emo_rec_sum / 7, 3)
    if act is not None:
        for i in range(len(act)):
            act_rec_sum += _weird_division(act[i][i], np.sum(act[i]))
        act_acc = round(act_rec_sum / 4, 3)
    if topic is not None:
        for i in range(len(topic)):
            topic_rec_sum += _weird_division(topic[i][i], np.sum(topic[i]))
        topic_acc = round(topic_rec_sum / 10, 3)
    return emo_acc, act_acc, topic_acc, phq_acc


def cal_f1(confmat, params):
    # calculate macro and micro-f1 score
    has_emo = params['has_emo']
    has_phq = params['has_phq']
    has_act = params['has_act']
    has_topic = params['has_topic']
    
    if has_emo and has_phq and has_act and has_topic:
        cm_emo = confmat[0]
        cm_act = confmat[1]
        cm_topic = confmat[2]
        cm_phq = confmat[3]
    elif has_phq and has_act:
        cm_act = confmat[0]
        cm_phq = confmat[1]
    elif has_phq and has_topic:
        cm_topic = confmat[0]
        cm_phq = confmat[1]
    elif has_emo and has_phq:
        cm_emo = confmat[0]
        cm_phq = confmat[1]
    elif has_phq:
        cm_phq = confmat
    ret = []

    #@Emotion F1
    if has_emo:
        cm = np.array(cm_emo)
        df = pd.DataFrame(cm, dtype=int)        
        # macro and micro with all class
        macro_rec = round((df.da.recall.sum()) / len(df.da.recall),4)
        macro_pre = round((df.da.precision.sum()) / len(df.da.precision),4)
        # macro_f1 = round((df.da.f1.sum()) / len(df.da.f1),4) #df.da.f1 gives f1 for each class
        f1 = 2 * macro_pre * macro_rec / (macro_rec + macro_pre)
        TP = df.da.TP.sum()
        FP = df.da.FP.sum()
        FN = df.da.FN.sum()
        micro_prec = TP / (TP+FP)
        micro_reca = TP / (TP+FN)
        micro_f1 = round(2*micro_prec*micro_reca / (micro_prec+micro_reca),4)
        # add accuracy
        acc = round(sum([df[i][i] for i in range(len(df))]) / df.to_numpy().sum(), 4)

        # macro and micro without neutre
        macro_f1_wo = round(df.da.f1[1:].sum() / len(df.da.f1[1:]),4)
        TP_wo = df.da.TP[1:].sum()
        FP_wo = df.da.FP[1:].sum()
        FN_wo = df.da.FN[1:].sum()
        micro_prec_wo = TP_wo / (TP_wo+FP_wo)
        micro_reca_wo = TP_wo / (TP_wo+FN_wo)
        
        micro_f1_wo = round(2*micro_prec_wo*micro_reca_wo / (micro_prec_wo+micro_reca_wo),4)
        ret.extend([f1, macro_pre, macro_rec, micro_f1, acc, macro_f1_wo, micro_f1_wo])

    #@Speech act F1
    if has_act:
        cm = np.array(cm_act)
        df = pd.DataFrame(cm, dtype=int)
        # macro and micro with all class
        # macro_f1 = round((df.da.f1.sum()) / len(df.da.f1),4) #df.da.f1 gives f1 for each class
        macro_rec = round((df.da.recall.sum()) / len(df.da.recall),4)
        macro_pre = round((df.da.precision.sum()) / len(df.da.precision),4)
        f1 = 2 * macro_pre * macro_rec / (macro_rec + macro_pre)
        TP = df.da.TP.sum()
        FP = df.da.FP.sum()
        FN = df.da.FN.sum()
        micro_prec = TP / (TP+FP)
        micro_reca = TP / (TP+FN)
        micro_f1 = round(2*micro_prec*micro_reca / (micro_prec+micro_reca),4)
        acc = round(sum([df[i][i] for i in range(len(df))]) / df.to_numpy().sum(), 4)
        ret.extend([f1, macro_pre, macro_rec, micro_f1, acc])

    #@Topic F1
    if has_topic:
        cm = np.array(cm_topic)
        df = pd.DataFrame(cm, dtype=int)
        # macro and micro with all class
        # macro_f1 = round((df.da.f1.sum()) / len(df.da.f1),4) #df.da.f1 gives f1 for each class
        macro_rec = round((df.da.recall.sum()) / len(df.da.recall),4)
        macro_pre = round((df.da.precision.sum()) / len(df.da.precision),4)
        f1 = 2 * macro_pre * macro_rec / (macro_rec + macro_pre)
        TP = df.da.TP.sum()
        FP = df.da.FP.sum()
        FN = df.da.FN.sum()
        micro_prec = TP / (TP+FP)
        micro_reca = TP / (TP+FN)
        micro_f1 = round(2*micro_prec*micro_reca / (micro_prec+micro_reca),4)
        acc = round(sum([df[i][i] for i in range(len(df))]) / df.to_numpy().sum(), 4)
        ret.extend([f1, macro_pre, macro_rec, micro_f1, acc])

    #@PHQ-9 F1
    if has_phq:
        cm = np.array(cm_phq)
        df = pd.DataFrame(cm, dtype=int)
        # macro and micro with all class
        # macro_f1 = round((df.da.f1.sum()) / len(df.da.f1),4) #df.da.f1 gives f1 for each class
        macro_rec = round((df.da.recall.sum()) / len(df.da.recall),4)
        macro_pre = round((df.da.precision.sum()) / len(df.da.precision),4)
        f1 = 2 * macro_pre * macro_rec / (macro_rec + macro_pre)
        TP = df.da.TP.sum()
        FP = df.da.FP.sum()
        FN = df.da.FN.sum()
        micro_prec = TP / (TP+FP)
        micro_reca = TP / (TP+FN)
        micro_f1 = round(2*micro_prec*micro_reca / (micro_prec+micro_reca),4)
        acc = round(sum([df[i][i] for i in range(len(df))]) / df.to_numpy().sum(), 4)
        ret.extend([f1, macro_pre, macro_rec, micro_f1, acc])
    return ret
    

def write_args2config(params):
    # take parser arguments to produce a json document, output is a string
    use_bert = str(params['use_bert'])[0].lower() + str(params['use_bert'])[1:]
    has_emo = str(params['has_emo'])[0].lower() + str(params['has_emo'])[1:]
    has_act = str(params['has_act'])[0].lower() + str(params['has_act'])[1:]
    has_topic = str(params['has_topic'])[0].lower() + str(params['has_topic'])[1:]
    has_phq = str(params['has_phq'])[0].lower() + str(params['has_phq'])[1:]
    has_phqbi = str(params['has_phqbi'])[0].lower() + str(params['has_phqbi'])[1:]
    orig_separation = str(params['orig_separation'])[0].lower() + str(params['orig_separation'])[1:]
    daic_resize = str(params['daic_resize'])[0].lower() + str(params['daic_resize'])[1:]
    del_ellie = str(params['del_ellie'])[0].lower() + str(params['del_ellie'])[1:]

    if params['use_bert']:
        turn_h = 768
        tokenizer = '{"type": "pretrained_transformer", "model_name": "bert-base-uncased"}'
        token_indexer = '''{"bert_tokens": {"type": "pretrained_transformer", "model_name": "bert-base-uncased"}}'''
        embedder = '''{
            "token_embedders": {
                "bert_tokens": {
                    "type": "pretrained_transformer",
                    "model_name": "bert-base-uncased",
                    "last_layer_only": true,
                    "train_parameters": false
                    }
                }
            }'''
        turn_encoder = '''{
            "type": "cls_pooler", 
            "embedding_dim": 768 
            }'''
        block_encoder = f'''{{
            "type": "gru",
            "input_size": 768,
            "hidden_size": {params['block_h']},
            "num_layers": {params['block_layer']},
            "dropout": {params['dropout']},
            "bidirectional": false
            }}'''
        
    else:
        turn_h = params['turn_h']
        tokenizer = '{"type": "spacy", "language": "en_core_web_sm"}'
        token_indexer = '''{"tokens": {"type": "single_id"}}'''
        embedder = '''{"token_embedders": {"tokens": {"type": "embedding", "embedding_dim": 128}}}'''
        turn_encoder = f'''{{
            "type": "lstm", 
            "input_size": 128, 
            "hidden_size": {turn_h}, 
            "num_layers": 1,
            "dropout": {params['dropout']},
            "bidirectional": true
            }}'''
        block_encoder = f'''{{
            "type": "gru",
            "input_size": {turn_h*2},
            "hidden_size": {params['block_h']},
            "num_layers": {params['block_layer']},
            "dropout": {params['dropout']},
            "bidirectional": false
            }}'''
        

    CONFIG = f"""
    {{
        "dataset_reader" : {{
            "type": "dialog_reader",
            "tokenizer": {tokenizer},
            "token_indexers":{token_indexer},
            "params":{{
                "encode_turns": false,
                "has_emo": {has_emo},
                "has_act": {has_act},
                "has_topic": {has_topic},
                "has_phq": {has_phq},
                "has_phqbi": {has_phqbi},
                "daic_resize": {daic_resize},
                "del_ellie": {del_ellie},
                "orig_separation": {orig_separation}
            }}
        }},  
        "train_data_path": "train",
        "validation_data_path": "dev",
        "model": {{
            "type": "hierarchical_classifier",
            "embedder": {embedder},
            "turn_encoder": {turn_encoder},
            "block_encoder": {block_encoder},
            "params":{{
                "has_emo": {has_emo},
                "has_act": {has_act},
                "has_topic": {has_topic},
                "has_phq": {has_phq},
                "has_phqbi": {has_phqbi},
                "use_bert": {use_bert}, 
                "lphq_coef": {params['lphq_coef']},
                "lemo_coef": {params['lemo_coef']},
                "lact_coef": {params['lact_coef']},
                "ltopic_coef": {params['ltopic_coef']},
                "cuda": {params['cuda']}
            }}
        }},
        "data_loader": {{
            "batch_size": {params['batchsize']},
            "shuffle": true
        }},
        "trainer": {{
            "optimizer": {{
                "type": "{params['optimizer']}",
                "lr": {params['lr']}
                }},
            "num_epochs": {params['epoch']},
            "cuda_device": {params['cuda']},
            "patience": 5
        }}
    }}
    """
    return CONFIG
