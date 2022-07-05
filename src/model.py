from allennlp.common.from_params import FromParams
import torch
from torch.nn.modules import Dropout
import torch.nn.functional as F
from torch import autograd
from typing import Dict, List, Iterable
from allennlp.models import Model
from allennlp.modules import TimeDistributed, TextFieldEmbedder, Seq2VecEncoder
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure
from allennlp.nn.util import get_text_field_mask
from allennlp.common import FromParams

@Model.register("hierarchical_classifier")
class HierarchicalClassification(Model, FromParams):

    # complicated way of doing it: better to put turn_encoder inside a curtom textfield embedder
    def __init__(self, vocab, embedder: TextFieldEmbedder, turn_encoder: Seq2VecEncoder, 
                block_encoder: Seq2VecEncoder,
                params: Dict = None
                ):
        super(HierarchicalClassification, self).__init__(vocab)
        self.text_embedder = embedder
        self.turn_encoder = TimeDistributed(turn_encoder) #reshapes the input to be `(batch_size * time_steps, [rest])`, then reshapes it back.
        self.block_encoder = block_encoder
        self.use_bert = params['use_bert']
        self.l_phq_coef = params['lphq_coef']
        self.l_emo_coef = params['lemo_coef']
        self.l_act_coef = params['lact_coef']
        self.l_topic_coef = params['ltopic_coef']
        self.has_emo = params['has_emo']
        self.has_act = params['has_act']
        self.has_topic = params['has_topic']
        self.has_phq = params['has_phq']
        self.has_phqbi = params['has_phqbi']
        # 2 classification layers, one for turn, one for block, transforms the output of Seq2VecEncoder into logits
        if self.use_bert:
            self.classif_layer_emo = torch.nn.Linear(in_features=self.text_embedder.get_output_dim(), out_features=7) #7 emotion[0-6]
            self.classif_layer_act = torch.nn.Linear(in_features=self.text_embedder.get_output_dim(), out_features=4) #4 speech act [1-4]
            # self.relu = torch.nn.ReLU()
        else:
            self.classif_layer_emo = torch.nn.Linear(in_features=self.turn_encoder._module.get_output_dim(), out_features=7)
            self.classif_layer_act = torch.nn.Linear(in_features=self.turn_encoder._module.get_output_dim(), out_features=4)
        
        self.classif_layer_topic = torch.nn.Linear(in_features=self.block_encoder.get_output_dim(), out_features=10) #topics [1-10]
        if self.has_phq and not self.has_phqbi:
            self.classif_layer_phq = torch.nn.Linear(in_features=self.block_encoder.get_output_dim(), out_features=5) #phq scores [0-4]
        elif self.has_phq and self.has_phqbi:
            self.classif_layer_phq = torch.nn.Linear(in_features=self.block_encoder.get_output_dim(), out_features=2) #phq scores [0,1]
        # 4 accuracies
        self.accuracy_emo = CategoricalAccuracy()
        self.accuracy_act = CategoricalAccuracy()
        self.accuracy_topic = CategoricalAccuracy()
        self.accuracy_phq = CategoricalAccuracy()
        self.f1_emo = FBetaMeasure(beta=1.0, average='macro')
        self.f1_act = FBetaMeasure(beta=1.0, average='macro')
        self.f1_topic = FBetaMeasure(beta=1.0, average='macro')
        self.f1_phq = FBetaMeasure(beta=1.0, average='macro')
        # 4 losses
        self._loss_emo = torch.nn.CrossEntropyLoss(ignore_index=-1)#when batch_size>1, would appear padded value (-1) in nb_block
        self._loss_act = torch.nn.CrossEntropyLoss(ignore_index=-1) 
        self._loss_topic = torch.nn.CrossEntropyLoss(ignore_index=-1) 
        self._loss_phq = torch.nn.CrossEntropyLoss(ignore_index=-1) 

        cuda = params['cuda']
        if cuda != -1:
            self.to(torch.device(f'cuda:{cuda}'))
        
    # @@ATTETNION: params in forward should have the same name with those in fileds when creating the data
    def forward(self,
                lines,
                label_emo = None, 
                label_act = None,
                label_topic = None,
                label_phq = None,
                ):
        # mask for each turn of each block of the batch: shape = (batch_size x nb_turns x max_tokens)
        mask = get_text_field_mask(lines, num_wrapping_dims=1)
        embedded_text = self.text_embedder(lines, num_wrapping_dims=1)
        
        # encoding turn
        if self.use_bert:
            embedded_text = embedded_text.masked_fill(~mask.unsqueeze(-1), 0.0)
            turn_h = embedded_text[:, :, 0, :]
        else:
            turn_h = self.turn_encoder(embedded_text, mask) 

        # encoding turn for dailydialog
        if self.has_emo:
            logits_emo = self.classif_layer_emo(turn_h) 
            probs_emo = F.softmax(logits_emo, dim=-1)
        if self.has_act:
            logits_act = self.classif_layer_act(turn_h)
            probs_act = F.softmax(logits_act, dim=-1)

        # encoding block for daic and dailydialog
        if self.has_phq or self.has_topic: 
            block_mask = mask.max(axis=2)[0]
            block_h = self.block_encoder(turn_h, block_mask) #[1,256]
            if self.has_phq:
                logits_phq = self.classif_layer_phq(block_h) #[1,5]
                probs_phq = F.softmax(logits_phq, dim=-1)
            if self.has_topic:
                logits_topic = self.classif_layer_topic(block_h) #[1,10]
                probs_topic = F.softmax(logits_topic, dim=-1)

        # multitask: emo, da, topic, phq
        if self.has_phq and self.has_emo and self.has_act and self.has_topic: #eatp
            output_dict = {"label_phq": label_phq,"probs_phq": probs_phq, "logits_phq": logits_phq, 
                "label_topic": label_topic,"probs_topic": probs_topic, "logits_topic": logits_topic, 
                "label_emo": label_emo,"probs_emo": probs_emo, "logits_emo": logits_emo,
                "label_act": label_act,"probs_act": probs_act, "logits_act": logits_act}
        elif self.has_phq and self.has_act: #ap
            output_dict = {"label_phq": label_phq,"probs_phq": probs_phq, "logits_phq": logits_phq, 
                "label_act": label_act,"probs_act": probs_act, "logits_act": logits_act
                }
        elif self.has_phq and self.has_topic: #tp
            output_dict = {"label_phq": label_phq,"probs_phq": probs_phq, "logits_phq": logits_phq, 
                "label_topic": label_topic,"probs_topic": probs_topic, "logits_topic": logits_topic
                }
        elif self.has_phq and self.has_emo: #ep
            output_dict = {"label_phq": label_phq,"probs_phq": probs_phq, "logits_phq": logits_phq, 
                "label_emo": label_emo,"probs_emo": probs_emo, "logits_emo": logits_emo
                }
        # monotask
        elif self.has_phq:#phq score
            output_dict = {"label_phq": label_phq, "probs_phq": probs_phq, "logits_phq": logits_phq}

        if self.has_emo:
            self.loss_emo = self._loss_emo(logits_emo.view(-1, 7), label_emo.long().view(-1))
            output_dict["loss_emo"] = self.loss_emo 
            if not set(label_emo.tolist()[0]) == {-1}:
                self.accuracy_emo(logits_emo, label_emo)
                self.f1_emo(predictions=logits_emo, gold_labels=label_emo) #need to mask , mask=mask[:,:,0]
        
        if self.has_act:
            self.loss_act = self._loss_act(logits_act.view(-1, 4), label_act.long().view(-1))
            output_dict["loss_act"] = self.loss_act
            if not set(label_act.tolist()[0]) == {-1}:
                self.accuracy_act(logits_act, label_act)
                self.f1_act(predictions=logits_act, gold_labels=label_act) #need to mask?

        if self.has_topic:
            self.loss_topic = self._loss_topic(logits_topic.view(-1, 10), label_topic.long().view(-1))
            output_dict["loss_topic"] = self.loss_topic
            if not set(label_topic.tolist()) == {-1}:
                self.accuracy_topic(logits_topic, label_topic)
                self.f1_topic(predictions=logits_topic, gold_labels=label_topic)

        if self.has_phq:
            self.loss_phq = self._loss_phq(logits_phq.view(-1, 2), label_phq.long().view(-1))
            output_dict["loss_phq"] = self.loss_phq
            if not set(label_phq.tolist()) == {-1}:
                self.accuracy_phq(logits_phq, label_phq)
                self.f1_phq(predictions=logits_phq, gold_labels=label_phq)

        if self.has_phq and self.has_emo and self.has_act and self.has_topic:
            output_dict["loss"] = self.l_emo_coef * self.loss_emo + self.l_phq_coef * self.loss_phq + self.l_topic_coef * self.loss_topic + self.l_act_coef * self.loss_act
        elif self.has_phq and self.has_topic:
            output_dict["loss"] = self.l_phq_coef * self.loss_phq + self.l_topic_coef * self.loss_topic
        elif self.has_emo and self.has_phq: 
            output_dict["loss"] = self.l_emo_coef * self.loss_emo + self.l_phq_coef * self.loss_phq
        elif self.has_act and self.has_phq: 
            output_dict["loss"] = self.l_act_coef * self.loss_act + self.l_phq_coef * self.loss_phq
        else:
            output_dict["loss"] = self.loss_phq
        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """for early stopping and model serialization, not required to implement metrics for a new model"""
        if self.has_phq and self.has_emo and self.has_act and self.has_topic:
            return_dict = {
                "accuracy_emo": self.accuracy_emo.get_metric(reset),
                "accuracy_act": self.accuracy_act.get_metric(reset),
                "accuracy_topic": self.accuracy_topic.get_metric(reset),
                "accuracy_phq": self.accuracy_phq.get_metric(reset),
            }
            return self.get_f_score(return_dict, ['emo', 'phq', 'act', 'topic'])
        elif self.has_phq and self.has_act:
            return_dict = {
                "accuracy_act": self.accuracy_act.get_metric(reset),
                "accuracy_phq": self.accuracy_phq.get_metric(reset),
            }
            return self.get_f_score(return_dict, ['phq', 'act'])
        elif self.has_phq and self.has_topic:
            return_dict = {
                "accuracy_topic": self.accuracy_topic.get_metric(reset),
                "accuracy_phq": self.accuracy_phq.get_metric(reset)
                }
            return self.get_f_score(return_dict, ['phq', 'topic'])
        elif self.has_emo and self.has_phq:
            return_dict = {
                "accuracy_emo": self.accuracy_emo.get_metric(reset),
                "accuracy_phq": self.accuracy_phq.get_metric(reset),
                "loss_emo": float(self.loss_emo.cpu().detach().numpy()), 
                "loss_phq": float(self.loss_phq.cpu().detach().numpy())
                }
            return self.get_f_score(return_dict, ['emo', 'phq'])
        elif self.has_phq:
            return_dict = {
                "accuracy_phq": self.accuracy_phq.get_metric(reset),
                'loss_phq': float(self.loss_phq.cpu().detach().numpy()),
            }
            return self.get_f_score(return_dict, ['phq'])


    def get_f_score(self, dict, lst_tgt, reset: bool = False)-> Dict[str, float]:
        tgt_dico = {'emo': self.f1_emo, 'phq': self.f1_phq, 'act': self.f1_act, 'topic': self.f1_topic}
        for tgt in lst_tgt:
            try:
                dict[f'f1_{tgt}'] = tgt_dico[tgt].get_metric(reset)['fscore']
            except:
                dict[f'f1_{tgt}'] = -1.0
        return dict

