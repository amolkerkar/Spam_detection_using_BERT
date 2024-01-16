import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast

model_path = 'E:/Projects/BERT/BERT/Data/SpamClassifierModel.pth'
mapping = {0: "Not spam", 1: "Spam"}
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
device = torch.device('cpu')

class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()

        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):

        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x

model = BERT_Arch(bert)
model = model.to(device)

# Load the model on the CPU
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

def classifier(smslist):
    classified = {}
    for text in smslist:
        tokens = tokenizer(text, return_tensors="pt")
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']

        with torch.no_grad():
            outputs = model(input_ids.to(device), mask=attention_mask.to(device))
            outputs = outputs.detach().cpu().numpy()

        probabilities = torch.exp(torch.tensor(outputs))
        predicted_class = torch.argmax(probabilities, dim=1).item()
        classified[text] = predicted_class

    print(classified)
    response = {"Not spam": [], "Spam": []}
    for sentence, flag in classified.items():
        response[mapping[flag]].append(sentence)
    print(response)
    return response
