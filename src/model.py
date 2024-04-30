from torch import nn
from transformers import BertForSequenceClassification, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class CustomModelForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        self.linear = nn.Linear(config.hidden_size, config.num_labels)
        if self.num_labels == 2:
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        hidden_state = outputs.last_hidden_state[:, 0, :]
        logits = self.activation(self.linear(hidden_state))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
