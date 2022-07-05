import json
import os
import pathlib
from typing import Dict, List, Iterable, Any
import torch
import torch.optim as optim
import argparse
import datetime

from allennlp.data import DataLoader #2.0
from allennlp.data.data_loaders import SimpleDataLoader, MultiProcessDataLoader #2.0
from allennlp.data.dataset_readers import DatasetReader, TextClassificationJsonReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, PretrainedTransformerIndexer, PretrainedTransformerMismatchedIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer, PretrainedTransformerTokenizer
from allennlp.models import Model
from allennlp.models.archival import archive_model, load_archive
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, BertPooler, ClsPooler, BagOfEmbeddingsEncoder, GruSeq2VecEncoder, LstmSeq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, PretrainedTransformerEmbedder, PretrainedTransformerMismatchedEmbedder
from allennlp.training.trainer import Trainer
from allennlp.training import GradientDescentTrainer
from allennlp.training.util import evaluate
from allennlp.common import Params

from constant import DAILY_DIR, DUAL_DIR, BERT_BASE
from utility import test_doc_res, calculate_balanced_acc, cal_f1, write_args2config

from dataset_reader import DialogReader
from model import HierarchicalClassification

def build_dataset_reader(params: dict) -> DatasetReader:
    if params['use_bert']:
        tokenizer = PretrainedTransformerTokenizer(model_name=BERT_BASE)
        indexer = {'bert_tokens': PretrainedTransformerIndexer(model_name=BERT_BASE)} 
    else:
        tokenizer = SpacyTokenizer(language="en_core_web_sm")
        indexer = {'tokens': SingleIdTokenIndexer()}
    reader = DialogReader(tokenizer=tokenizer, token_indexers=indexer, params=params)
    return reader
    
def read_data(reader: DatasetReader, outf: str=None, MODE: str='train-dev') -> Iterable[Instance]:
    if MODE == 'train-dev':
        train_data = list(reader.read(file_path='train'))
        dev_data = list(reader.read(file_path='dev'))
        return train_data, dev_data
    elif MODE == 'test':
        train_data = list(reader.read(file_path='train'))
        test_data = list(reader.read(file_path='test'))
        return train_data, test_data
    
def build_vocab(params: dict, instances: Iterable[Instance]) -> Vocabulary:
    if params['use_bert']: 
        return Vocabulary()
    else:
        return Vocabulary.from_instances(instances)

def build_model(vocab: Vocabulary, params: dict)-> Model:
    # print("Building the model...")    
    if params['use_bert']:
        token_embedding = PretrainedTransformerEmbedder(
            model_name=BERT_BASE,
            last_layer_only=True,
            train_parameters=params['bert_finetune'],
        )
        word_embedder = BasicTextFieldEmbedder(token_embedders={"bert_tokens": token_embedding})
        turn_encoder = ClsPooler(embedding_dim=768) 
        block_encoder = GruSeq2VecEncoder(input_size=768, hidden_size=params['block_h'], num_layers=params['block_layer'], \
                                        dropout=params['dropout'], bidirectional=False)
    else: # vanilla embedding
        vocab_size = vocab.get_vocab_size("tokens")
        token_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=128)
        word_embedder = BasicTextFieldEmbedder(token_embedders={"tokens": token_embedding})
        turn_encoder = LstmSeq2VecEncoder(input_size=128, hidden_size=params['turn_h'], num_layers=1, dropout=params['dropout'],
                                        bidirectional=True)
        block_encoder = GruSeq2VecEncoder(input_size=params['turn_h']*2, hidden_size=params['block_h'], num_layers=params['block_layer'], \
                                        dropout=params['dropout'], bidirectional=False)
    return HierarchicalClassification(vocab, word_embedder, turn_encoder, block_encoder, params=params)

def build_data_loaders(
    train_data: List[Instance],
    dev_test_data: List[Instance], #dev or test according to MODE
    batch_size: int
    ) -> [DataLoader, DataLoader]:
    train_loader = SimpleDataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_test_loader = SimpleDataLoader(dev_test_data, batch_size=batch_size)
    return train_loader, dev_test_loader

def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    params: dict=None
    ) -> Trainer:
    optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr=params['lr'])
    
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        patience=100,
        validation_metric="+f1_phq", #-loss_phq
        num_epochs=params['epoch'],
        optimizer=optimizer,
        cuda_device=params['cuda'],
    )
    return trainer

def run_training_loop(params: dict, outf: str=None, serialdir: str='savedmodel', config:str=''
    ) -> [Model, Dict, DataLoader]:
    
    dataset_reader = build_dataset_reader(params)

    train_data, dev_data = read_data(dataset_reader, outf, 'train-dev')

    vocab = build_vocab(params, train_data)
    model = build_model(vocab, params)

    train_loader, dev_loader = build_data_loaders(train_data, dev_data, batch_size=params['batchsize'])

    train_loader.index_with(vocab)
    dev_loader.index_with(model.vocab)

    trainer = build_trainer(model, serialdir, train_loader, dev_loader, params)
    metrics = trainer.train()

    config_file = os.path.join(serialdir, "config.json")
    vocabulary_dir = os.path.join(serialdir, "vocabulary")
    weights_file = os.path.join(serialdir, "weights.th")

    os.makedirs(serialdir, exist_ok=True)
    model_params = Params(json.loads(config))
    model_params.to_file(config_file)
    vocab.save_to_files(vocabulary_dir)
    torch.save(model.state_dict(), weights_file)

    archive_model(
        serialization_dir=serialdir, 
        weights='best.th'
        )
    return model, metrics, dev_loader

def run_test(model: Model, params: dict, outf: str) -> DataLoader:
    dataset_reader = build_dataset_reader(params)

    train_data, dev_test_data = read_data(dataset_reader, outf, 'test')

    vocab = build_vocab(params, train_data)

    train_loader, test_loader = build_data_loaders(train_data, dev_test_data, batch_size=params['batchsize'])

    train_loader.index_with(vocab)
    test_loader.index_with(model.vocab)

    return test_loader


def test_results(model: Model, test_loader: DataLoader, cuda:int, preddir:str, MODE:str, params: dict) -> Dict[str, Any]:
    if MODE == 'train-dev':
        pred_f = os.path.join(preddir, f"train-dev_detail")
    elif MODE == 'test':
        pred_f = os.path.join(preddir, f"test_detail")
    
    results = evaluate(
        model=model, 
        data_loader=test_loader,
        cuda_device=cuda,
        output_file=None,
        predictions_output_file=pred_f
        )
    details = test_doc_res(pred_f, params)
    return results, details
    

if __name__ == "__main__":
    datadir = DAILY_DIR
    codedir = DUAL_DIR

    parser = argparse.ArgumentParser(description='Dual encoding to test emotion and PHQ level classification for dailydialog and daic respectively. Configs found in `constant.py` file, modification if needed.')
    parser.add_argument('--MODE', dest='MODE', default='train-dev', type=str, help="Choose from train-dev or test.")
    parser.add_argument('--configID', dest='configID', type=str, help="Name of the test configuration id, see google sheet optuna-tune.")
    parser.add_argument('--maxi', dest='maxi', default='emo-phq', type=str, help="maximize objectives: emo-phq or emo or phq. Goes along with has_emo, has_phq.")
    parser.add_argument('--use_bert', dest='use_bert', default=False, action='store_true', help="Use pretrained embedding from bert. By default=False, use random iniitalised embeddings.")
    parser.add_argument('--bert_finetune', dest='bert_finetune', default=False, action='store_true', help="Use pretrained embedding bert and train parameters while training.")
    parser.add_argument('--has_emo', dest='has_emo', default=False, action='store_true', help="predict emotion in dialy.")
    parser.add_argument('--has_topic', dest='has_topic', default=False, action='store_true', help="predict topic.")
    parser.add_argument('--has_act', dest='has_act', default=False, action='store_true', help="predict speech act.")
    parser.add_argument('--has_phq', dest='has_phq', default=False, action='store_true', help="predict depression in daic, doc level")
    parser.add_argument('--has_phqbi', dest='has_phqbi', default=False, action='store_true', help="predict binarized depression level in daic")
    parser.add_argument('--cuda', dest='cuda', default=-1, type=int, help='Choose cuda if available.')
    parser.add_argument('--batchsize', dest='batchsize', default=16, type=int, help='Param for dataloader, each time load n instance. By default=1.')
    parser.add_argument('--epoch', dest='epoch', default=1, type=int, help='Epoch for train. By default=1, suggest at least 10 while training.')
    parser.add_argument('--lphq_coef', dest='lphq_coef', default=1.0, type=float, help="""Coefficient for loss phq""")
    parser.add_argument('--lemo_coef', dest='lemo_coef', default=1.0, type=float, help="""Coefficient for loss emotion""")
    parser.add_argument('--lact_coef', dest='lact_coef', default=1.0, type=float, help="""Coefficient for loss dialogue act""")
    parser.add_argument('--ltopic_coef', dest='ltopic_coef', default=1.0, type=float, help="""Coefficient for loss topic""")
    parser.add_argument('--turn_h', dest='turn_h', default=128, type=int, help="Hidden size for turn encoder. Note that word embeddings is 768 for bert. By default is 128.")
    parser.add_argument('--block_h', dest='block_h', default=256, type=int, help="Hidden size for block encoder. By default is 256.")
    parser.add_argument('--block_layer', dest='block_layer', default=1, type=int, help="NB of layers for block encoder. By default is 1.")
    parser.add_argument('--dropout', dest='dropout', default=0.25, type=float, help="The dropout rate for block encoder and doc encoder. By default is 0.25. Choose between [0.0, 0.8]")
    parser.add_argument('--lr', dest='lr', default=0.001, type=float, help='Learning rate, by default=0.001')
    parser.add_argument('--optimizer', dest='optimizer', default='Adam', type=str, help="Name of the optimizer. Choose from adam and sgd.")
    parser.add_argument('--orig_separation', dest='orig_separation', default=False, action='store_true', help="Use dailydialog original separation for train/dev/test.")
    args = parser.parse_args()

    configID = args.configID
    maxi = args.maxi

    params = {}
    params['MODE'] = args.MODE
    params['encode_turns'] = False
    params['use_bert'] = args.use_bert
    params['bert_finetune'] = args.bert_finetune
    params['has_emo'] = args.has_emo
    params['has_act'] = args.has_act
    params['has_topic'] = args.has_topic
    params['has_phq'] = args.has_phq
    params['has_phqbi'] = args.has_phqbi
    params['cuda'] = args.cuda
    params['batchsize'] = args.batchsize
    params['epoch'] = args.epoch
    params['lphq_coef'] = args.lphq_coef
    params['lemo_coef'] = args.lemo_coef
    params['lact_coef'] = args.lact_coef
    params['ltopic_coef'] = args.ltopic_coef
    params['turn_h'] = args.turn_h
    params['block_h'] = args.block_h
    params['block_layer'] = args.block_layer
    params['dropout'] = args.dropout
    params['lr'] = args.lr
    params['optimizer'] = args.optimizer
    params['orig_separation'] = args.orig_separation

    # check for cuda availability
    if torch.cuda.is_available():
        cuda_device = params['cuda']
    else:
        cuda_device = -1

    # write args into json config
    CONFIG = write_args2config(params)

    # write logs
    dt = datetime.datetime.today()
    datedir = 'xx-yy'
    if params['use_bert'] and params['bert_finetune']:
        datedir += '-bert-ft'
    seeddir = codedir + f"log/{datedir}/{maxi}/{configID}/" 
    serialdir = codedir + f'models/{datedir}/{maxi}/{configID}/'
    pathlib.Path(seeddir).mkdir(parents=True, exist_ok=True)
    
    conf_mat = ''
    otherscores = ''
    if params['MODE'] == 'train-dev':
        resf = os.path.join(seeddir, f"train-dev")

        with open(resf, 'w') as f:
            config = ''
            for (key, value) in params.items():
                if key == 'turn_h' and params['use_bert']:
                    value = 768
                config = config + key + ': ' + str(value) + '\n'
            print(f'PARAMS: \n{config}', file=f)

            model, train_metrics, dev_loader = run_training_loop(params, outf=f, serialdir=serialdir, config=CONFIG)
            print('\nRESULTS:', file=f)
            print(train_metrics, file=f)
            
            dev_metrics, conf_mat = test_results(model, dev_loader, cuda=cuda_device, preddir=seeddir, MODE='train-dev', params=params)
            print(dev_metrics, file=f)
            print(conf_mat, file=f)

    elif params['MODE'] == 'test':
        resf = os.path.join(seeddir, f"test")
        
        with open(resf, 'w') as f:
            archive = load_archive(
                archive_file=os.path.join(serialdir, "model.tar.gz"),
                cuda_device=cuda_device
                )
            
            test_loader = run_test(model=archive.model, params=params, outf=f)

            test_metrics, conf_mat = test_results(archive.model, test_loader, cuda=cuda_device, preddir=seeddir, MODE='test', params=params)
            print(test_metrics, file=f)
            print(conf_mat, file=f)

    # add balanced accuracy
    bal_emo, bal_act, bal_topic, bal_phq = calculate_balanced_acc(conf_mat, params)    
    otherscores += 'Balanced accuracy (turn->emotion, da, block->topic, phq): '+ str((bal_emo, bal_act, bal_topic, bal_phq)) + '\n'

    #@calculate F1 and store in train-dev or test result file
    if params['has_emo'] and params['has_phq'] and params['has_act'] and params['has_topic']:
        emo_macrof1, emo_macro_pre, emo_macro_rec, emo_microf1, emo_acc, emo_macrof1_wo, emo_microf1_wo, act_macrof1, act_macro_pre, act_macro_rec, act_microf1, act_acc, top_macrof1, top_macro_pre, top_macro_rec, top_microf1, top_acc, phq_macrof1, phq_macro_pre, phq_macro_rec, phq_microf1, phq_acc = cal_f1(conf_mat, params)
        otherscores += f'Emotion F1\n   - Macro-F1-w-all: {emo_macrof1}\n   - Macro-Precision-w-all: {emo_macro_pre}\n   - Macro-Recall-w-all: {emo_macro_rec}\n   - Micro-F1-w-all: {emo_microf1}\n   - Acc-w-all: {emo_acc}\n   - Macro-F1-wo-neutral: {emo_macrof1_wo}\n   - Micro-F1-wo-neutral: {emo_microf1_wo}\n'
        otherscores += f'Speech act F1\n   - Macro-F1-w-all: {act_macrof1}\n   - Macro-Precision-w-all: {act_macro_pre}\n   - Macro-Recall-w-all: {act_macro_rec}\n   - Micro-F1-w-all: {act_microf1}\n   - Acc-w-all: {act_acc}\n'
        otherscores += f'Topic F1\n   - Macro-F1-w-all: {top_macrof1}\n   - Macro-Precision-w-all: {top_macro_pre}\n   - Macro-Recall-w-all: {top_macro_rec}\n   - Micro-F1-w-all: {top_microf1}\n   - Acc-w-all: {top_acc}\n'
        otherscores += f'PHQ F1\n   - Macro-F1-w-all: {phq_macrof1}\n   - Macro-Precision-w-all: {phq_macro_pre}\n   - Macro-Recall-w-all: {phq_macro_rec}\n   - Micro-F1-w-all: {phq_microf1}\n  - Acc-w-all: {phq_acc}\n'

    elif params['has_phq'] and params['has_act']:
        act_macrof1, act_macro_pre, act_macro_rec, act_microf1, act_acc, phq_macrof1, phq_macro_pre, phq_macro_rec, phq_microf1, phq_acc = cal_f1(conf_mat, params)
        otherscores += f'Speech act F1\n   - Macro-F1-w-all: {act_macrof1}\n   - Macro-Precision-w-all: {act_macro_pre}\n   - Macro-Recall-w-all: {act_macro_rec}\n   - Micro-F1-w-all: {act_microf1}\n   - Acc-w-all: {act_acc}\n'
        otherscores += f'PHQ F1\n   - Macro-F1-w-all: {phq_macrof1}\n   - Macro-Precision-w-all: {phq_macro_pre}\n   - Macro-Recall-w-all: {phq_macro_rec}\n   - Micro-F1-w-all: {phq_microf1}\n  - Acc-w-all: {phq_acc}\n'
    
    elif params['has_phq'] and params['has_topic']:
        top_macrof1, top_macro_pre, top_macro_rec, top_microf1, top_acc, phq_macrof1, phq_macro_pre, phq_macro_rec, phq_microf1, phq_acc = cal_f1(conf_mat, params)
        otherscores += f'Topic F1\n   - Macro-F1-w-all: {top_macrof1}\n   - Macro-Precision-w-all: {top_macro_pre}\n   - Macro-Recall-w-all: {top_macro_rec}\n   - Micro-F1-w-all: {top_microf1}\n   - Acc-w-all: {top_acc}\n'
        otherscores += f'PHQ F1\n   - Macro-F1-w-all: {phq_macrof1}\n   - Macro-Precision-w-all: {phq_macro_pre}\n   - Macro-Recall-w-all: {phq_macro_rec}\n   - Micro-F1-w-all: {phq_microf1}\n  - Acc-w-all: {phq_acc}\n'
    
    elif params['has_emo'] and params['has_phq']:
        emo_macrof1, emo_macro_pre, emo_macro_rec, emo_microf1, emo_acc, emo_macrof1_wo, emo_microf1_wo, phq_macrof1, phq_macro_pre, phq_macro_rec, phq_microf1, phq_acc = cal_f1(conf_mat, params)
        otherscores += f'Emotion F1\n   - Macro-F1-w-all: {emo_macrof1}\n   - Macro-Precision-w-all: {emo_macro_pre}\n   - Macro-Recall-w-all: {emo_macro_rec}\n   - Micro-F1-w-all: {emo_microf1}\n   - Acc-w-all: {emo_acc}\n   - Macro-F1-wo-neutral: {emo_macrof1_wo}\n   - Micro-F1-wo-neutral: {emo_microf1_wo}\n'
        otherscores += f'PHQ F1\n   - Macro-F1-w-all: {phq_macrof1}\n   - Macro-Precision-w-all: {phq_macro_pre}\n   - Macro-Recall-w-all: {phq_macro_rec}\n   - Micro-F1-w-all: {phq_microf1}\n  - Acc-w-all: {phq_acc}\n'
    
    elif params['has_phq']:
        phq_macrof1, phq_macro_pre, phq_macro_rec, phq_microf1, phq_acc = cal_f1(conf_mat, params)
        otherscores += f'PHQ F1\n   - Macro-F1-w-all: {phq_macrof1}\n   - Macro-Precision-w-all: {phq_macro_pre}\n   - Macro-Recall-w-all: {phq_macro_rec}\n   - Micro-F1-w-all: {phq_microf1}\n  - Acc-w-all: {phq_acc}\n'

    with open(resf, 'a') as f:
        print(otherscores, file=f)
    print(f'\nFINISH. Results stored in {resf}.')
