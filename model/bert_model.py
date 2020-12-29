import torch
import torch.nn as nn
from transformers import BertModel


class bert_classifi(nn.Module):
    def __init__(self, args):
        super(bert_classifi, self).__init__()

        self.output_hidden_states = args.output_hidden_states
        self.use_bert_dropout = args.use_bert_dropout

        self.bert_model = BertModel.from_pretrained(args.bert_path)
        for param in self.bert_model.parameters():
            param.requires_grad = True

        self.fc = nn.Linear(args.bert_dim*2, args.class_num)

        self.lstm = nn.LSTM(
            args.bert_dim, args.lstm_hidden_dim, num_layers=1, bidirectional=args.bilstm, batch_first=True
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.bert_dropout = nn.Dropout(args.bert_dropout)
        self.fc_dropout = nn.Dropout(args.fc_dropout)

    def forward(self, input_id, token_type_ids, attention_mask):
        word_vec, sen_vec, hidden_states = self.bert_model(
            input_ids=input_id, token_type_ids=token_type_ids, attention_mask=attention_mask,
            output_hidden_states=self.output_hidden_states
        )
        if self.use_bert_dropout:
            word_vec = self.bert_dropout(word_vec)
        avg_feature = self.avg_pool(word_vec.permute(0, 2, 1)).squeeze(-1)
        max_feature = self.max_pool(word_vec.permute(0, 2, 1)).squeeze(-1)

        feature = torch.cat((avg_feature, max_feature), dim=-1)
        feature = self.fc_dropout(feature)

        out = self.fc(feature)
        return out
