import torch
from fairseq.models import register_model, register_model_architecture
import torch.nn as nn
from transformers import BertModel, AutoConfig


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # config = AutoConfig.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        for param in self.bert.parameters():
            param.requires_grad = True

        self.dense = nn.Sequential(nn.Linear(768, 128),
                                   nn.Tanh(),
                                   nn.Linear(128, 3))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids_list, token_type_ids_list, attention_mask_list, labels_list):
        outputs = []
        for input_ids, token_type_ids, attention_mask, labels in zip(input_ids_list, token_type_ids_list,
                                                                     attention_mask_list, labels_list):
            # bert_output
            bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            bert_cls_hidden_state = bert_output.pooler_output

            linear_output = self.dense(bert_cls_hidden_state)

            softmax_output = nn.functional.log_softmax(linear_output, -1)
            outputs.append(softmax_output)

        # 计算损失
        loss = self.criterion(outputs[0], labels_list[0])

        p_mixture = torch.clamp(sum(outputs) / float(len(outputs)), 1e-7, 1).log()
        part = 0
        for i in outputs[1:]:
            part += nn.functional.kl_div(p_mixture, outputs[i], reduction='batchmean')

        if len(outputs)>1:
            loss += 8 * (nn.functional.kl_div(p_mixture, outputs[0], reduction='batchmean') +
                     part) / float(len(outputs))

        _, predict = torch.max(outputs[0].data, 1)

        return loss, predict, outputs[0]

# @register_model_architecture('classifier', 'base_arch')
# def classifier_args(args):
#     transformer_iwslt_de_en(args)
