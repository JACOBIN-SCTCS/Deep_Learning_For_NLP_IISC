#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from bs4 import BeautifulSoup
import string
import spacy
import jsonlines
import json
import re
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence,pad_sequence
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler,DataLoader,Subset
from torchtext.vocab import GloVe
from tqdm import tqdm
import io
from spacy.language import Language
from spacy.tokens import Doc
from transformers import DistilBertTokenizer, DistilBertModel

# SENTENCE_SPLITTING_USED; whether to use the splitting of reviews into sentences.
EMBED_DIM = 300
HIDDEN_DIM = 256
ATTENTION_DIM = 256

PATIENCE_PARAMETER = 7
VALIDATION_LOSS_COMPUTE_STEP = 1
EXPAND_CONTRACTIONS = True


device_cpu = torch.device('cpu')
device_fast = torch.device('cpu')

USE_PRETRAINED_MODEL= True

if torch.has_mps:
    device_fast = torch.device('mps')
elif torch.has_cuda:
    device_fast = torch.device('cuda')

#torch.manual_seed(0)
#np.random.seed(0)
glove = GloVe()
torch.cuda.empty_cache()
print(torch.cuda.is_available())


# In[2]:


def expand_contractions_text(text):    
    flags = re.IGNORECASE | re.MULTILINE
    text = re.sub(r'`', "'", text, flags = flags)
    ## starts / ends with '
    text = re.sub(
        r"(\s|^)'(aight|cause)(\s|$)",
        '\g<1>\g<2>\g<3>',
        text, flags = flags
    )
    text = re.sub(
        r"(\s|^)'t(was|is)(\s|$)", r'\g<1>it \g<2>\g<3>',
        text,
        flags = flags
    )
    text = re.sub(
        r"(\s|^)ol'(\s|$)",
        '\g<1>old\g<2>',
        text, flags = flags
    )
        
    text = re.sub(r"\b(aight)\b", 'alright', text, flags = flags)
    text = re.sub(r'\bcause\b', 'because', text, flags = flags)
    text = re.sub(r'\b(finna|gonna)\b', 'going to', text, flags = flags)
    text = re.sub(r'\bgimme\b', 'give me', text, flags = flags)
    text = re.sub(r"\bgive'n\b", 'given', text, flags = flags)
    text = re.sub(r"\bhowdy\b", 'how do you do', text, flags = flags)
    text = re.sub(r"\bgotta\b", 'got to', text, flags = flags)
    text = re.sub(r"\binnit\b", 'is it not', text, flags = flags)
    text = re.sub(r"\b(can)(not)\b", r'\g<1> \g<2>', text, flags = flags)
    text = re.sub(r"\bwanna\b", 'want to', text, flags = flags)
    text = re.sub(r"\bmethinks\b", 'me thinks', text, flags = flags)
    text = re.sub(r"\bo'er\b", r'over', text, flags = flags)
    text = re.sub(r"\bne'er\b", r'never', text, flags = flags)
    text = re.sub(r"\bo'?clock\b", 'of the clock', text, flags = flags)
    text = re.sub(r"\bma'am\b", 'madam', text, flags = flags)
    text = re.sub(r"\bgiv'n\b", 'given', text, flags = flags)
    text = re.sub(r"\be'er\b", 'ever', text, flags = flags)
    text = re.sub(r"\bd'ye\b", 'do you', text, flags = flags)
    text = re.sub(r"\be'er\b", 'ever', text, flags = flags)
    text = re.sub(r"\bd'ye\b", 'do you', text, flags = flags)
    text = re.sub(r"\bg'?day\b", 'good day', text, flags = flags)
    text = re.sub(r"\b(ain|amn)'?t\b", 'am not', text, flags = flags)
    text = re.sub(r"\b(are|can)'?t\b", r'\g<1> not', text, flags = flags)
    text = re.sub(r"\b(let)'?s\b", r'\g<1> us', text, flags = flags)
    text = re.sub(r"\by'all'dn't've'd\b", 'you all would not have had', text, flags = flags)
    text = re.sub(r"\by'all're\b", 'you all are', text, flags = flags)
    text = re.sub(r"\by'all'd've\b", 'you all would have', text, flags = flags)
    text = re.sub(r"(\s)y'all(\s)", r'\g<1>you all\g<2>', text, flags = flags)  
    text = re.sub(r"\b(won)'?t\b", 'will not', text, flags = flags)
    text = re.sub(r"\bhe'd\b", 'he had', text, flags = flags)
    text = re.sub(r"\b(I|we|who)'?d'?ve\b", r'\g<1> would have', text, flags = flags)
    text = re.sub(r"\b(could|would|must|should|would)n'?t'?ve\b", r'\g<1> not have', text, flags = flags)
    text = re.sub(r"\b(he)'?dn'?t'?ve'?d\b", r'\g<1> would not have had', text, flags = flags)
    text = re.sub(r"\b(daren|daresn|dasn)'?t", 'dare not', text, flags = flags)
    text = re.sub(r"\b(he|how|i|it|she|that|there|these|they|we|what|where|which|who|you)'?ll\b", r'\g<1> will', text, flags = flags)
    text = re.sub(r"\b(everybody|everyone|he|how|it|she|somebody|someone|something|that|there|this|what|when|where|which|who|why)'?s\b", r'\g<1> is', text, flags = flags)
    text = re.sub(r"\b(I)'?m'a\b", r'\g<1> am about to', text, flags = flags)
    text = re.sub(r"\b(I)'?m'o\b", r'\g<1> am going to', text, flags = flags)
    text = re.sub(r"\b(I)'?m\b", r'\g<1> am', text, flags = flags)
    text = re.sub(r"\bshan't\b", 'shall not', text, flags = flags)
    text = re.sub(r"\b(are|could|did|does|do|go|had|has|have|is|may|might|must|need|ought|shall|should|was|were|would)n'?t\b", r'\g<1> not', text, flags = flags)
    text = re.sub(r"\b(could|had|he|i|may|might|must|should|these|they|those|to|we|what|where|which|who|would|you)'?ve\b", r'\g<1> have', text, flags = flags)
    text = re.sub(r"\b(how|so|that|there|these|they|those|we|what|where|which|who|why|you)'?re\b", r'\g<1> are', text, flags = flags)
    text = re.sub(r"\b(I|it|she|that|there|they|we|which|you)'?d\b", r'\g<1> had', text, flags = flags)
    text = re.sub(r"\b(how|what|where|who|why)'?d\b", r'\g<1> did', text, flags = flags)    
    return text


class ExpandContractionsClass:
    def __init__(self, nlp: Language):
        self.nlp = nlp
    
    def __call__(self,doc: Doc):
        text = doc.text
        return self.nlp.make_doc(expand_contractions_text(text))
    
@Language.factory("expand_contractions_component")
def create_expand_contractions_component(nlp : Language, name: str):
    return ExpandContractionsClass(nlp)


# In[3]:


nlp = spacy.load('en_core_web_sm')
if EXPAND_CONTRACTIONS:
    nlp.add_pipe("expand_contractions_component",before='tagger')


# In[4]:


def preprocess_text(text):    
    
    text = re.sub(r'<br /><br />',"$$",text)
    text = BeautifulSoup(text,'lxml').get_text().strip()
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = ' '.join(re.findall(r"[\w']+|[.,!;/\"]", text))
    
    new_text = []
    for word in text.split():
        if word == '':
            continue
        new_text.append(word)
    
    text = ' '.join(new_text)
    words = nlp(text)
    text =  " ".join([token.text for token in words if not token.is_punct or token.text=='/' or token.text=="\"" or token.text=="."]).strip()
    
    new_sents = []
    for sent in text.split("."):
        sent = sent.strip()
        if sent!='' and len(sent)>1:
        
            new_sents.append(sent)

    text = ' . '.join(new_sents)

    new_words = []
    for word in text.split(" "):
        #if word == 'n\'t':
        #    if len(new_words) > 1:
        #        new_words[-1] = new_words[-1] + word
        #    else:
        #        new_words.append(word)
        if word == '\'s':
            if len(new_words) > 1:
                new_words[-1] = new_words[-1] + word
        else:
            new_words.append(word)
            
    text = " ".join(new_words)
    return text


# In[5]:


# preprocess the training data which was given for Assignment 2
def process_assignment2_training_data():
    preprocessed_dataset = []
    train_dataset_labels = []
    with open("./Train dataset.csv") as csvfile:
        csvFile = csv.reader(csvfile)
        next(csvFile)
        json_writer = jsonlines.open('processed_dataset.jsonl','w')

        for line in csvFile:
            processed_text = preprocess_text(line[0])
            label = 1.0 if line[1] == 'positive' else 0.0
            train_dataset_labels.append(label)
            json_writer.write({"text":processed_text,"label":label})
            preprocessed_dataset.append({"text":processed_text,"label":label})
    
        json_writer.close()


#process_assignment2_training_data()


# In[6]:


preprocessed_dataset = []
train_dataset_labels = []

TRAIN_FILE_NAME = './processed_dataset.jsonl'

with open(TRAIN_FILE_NAME ,encoding='utf-8') as f:
#with open('processed_dataset.jsonl',encoding='utf-8') as f:
    for line in f:
        sample = json.loads(line)
        train_dataset_labels.append(sample['label'])
        preprocessed_dataset.append(sample)
      
train_dataset_labels = np.array(train_dataset_labels)


# In[7]:


processed_dataset = []

for review in preprocessed_dataset:
    #embedding,length = getWordEmbeddingforText(review['text'])
    #processed_dataset.append({'text': embedding,'length': length,'label' : review['label']})
    processed_dataset.append({'text' : review['text'], 'label' : review['label']})
 


# In[8]:


class ReviewDataSet(Dataset):
    def __init__(self,data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

dataset = ReviewDataSet(processed_dataset)


# In[9]:


# Train and Validation split and an equal distriubition of classes

berttokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

train_idx,valid_idx = train_test_split(np.arange(train_dataset_labels.shape[0]), 
    test_size=0.2,
    shuffle= True,
    stratify= train_dataset_labels,
    random_state=0
)

'''def collate_function(batch_data):
    inputs = [b['text'] for b in batch_data]
    lengths = [b['length'] for b in batch_data]
    labels = torch.tensor([b['label'] for b in batch_data])

    labels = labels.unsqueeze(1)
    inputs = pad_sequence(inputs,batch_first=True)
    return  {'input' : inputs , 'lengths': lengths , 'labels' : labels }'''

def collate_function(batch_data):

    inputs = [b['text'] for b in batch_data]

    if USE_PRETRAINED_MODEL:
        ids = berttokenizer(inputs,padding=True,truncation=True,return_tensors="pt")
    else:
        ids = berttokenizer(inputs,padding=True,truncation=False,return_tensors="pt")

    lengths = torch.sum(ids['attention_mask'],dim=1).tolist()
    labels = torch.tensor([b['label'] for b in batch_data])
    return {'input' : ids['input_ids'] ,'lengths' : lengths , 'attention_mask' : ids['attention_mask'] ,'labels' : labels }
    

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
train_dataloader = DataLoader(dataset,64,sampler=train_sampler,collate_fn=collate_function)
valid_dataloader = DataLoader(dataset,64,sampler=valid_sampler,collate_fn=collate_function)


# In[10]:


batch_data = next(iter(train_dataloader))


# In[11]:


class PretrainedModel(nn.Module):

    def __init__(self, dropout = 0.3):
        super().__init__()

        self.bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.fc1 = nn.Linear(768,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,1)
        self.sigmoid = nn.Sigmoid()


    def freeze_weights(self):
        for param in self.bert_model.parameters():
            param.requires_grad_(False)


    def forward(self,inp,attention_mask):

        output = self.bert_model(input_ids=inp,attention_mask=attention_mask,return_dict=False)[0]
        
        out = output[:,0,:]
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))
        return out


# In[12]:


pre_model = PretrainedModel()
pre_model.freeze_weights()
out = pre_model(batch_data['input'],batch_data['attention_mask'])


# In[ ]:


from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout = 0.3, max_seq_length = 5000):
        
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros((max_seq_length,d_model))

        for i in range(max_seq_length):
            for j in range(d_model//2):
                pe[i,2*j] = math.sin(i/(10000.0**(2*j/d_model)))
                pe[i,2*j+1] = math.cos(i/(10000.0**(2*j/d_model)))

        self.register_buffer('pe', pe)
    
    def forward(self,x,x_mask):
        
        for i in range(len(x_mask)):
            x[i,:x_mask[i],:] = x[i,:x_mask[i],:] + self.pe[:x_mask[i],:] 

        return self.dropout(x)


# In[ ]:


class TransformerEncoderModel(nn.Module):

    def __init__(self,n_tokens=30522, d_model = 512, nhead=8, dim_feed_forward = 2048, nlayers=6, dropout = 0.3,fc_dropout_p=0.3, device_train = device_fast):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(d_model,nhead,dim_feed_forward,batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers,nlayers)
        self.device_train = device_train
       
        self.embedding_layer = nn.Embedding(n_tokens,d_model)
        self.d_model = d_model


        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,1)

        self.fc_dropout = nn.Dropout(fc_dropout_p)
        self.sigmoid = nn.Sigmoid()


     
    def create_src_mask(self,padding_mask):
        
        attention_mask = torch.zeros((len(padding_mask),max(padding_mask)),dtype=torch.bool)
        for i in range(len(padding_mask)):
            attention_mask[i,padding_mask[i]:] = True
        return attention_mask

    def init_weights(self):
        initrange = 0.1
        self.embedding_layer.weight.data.uniform_(-initrange, initrange)


    def forward(self, inp, inp_mask):
        
        # inp = [ batch_size, max_seq_length]
        # inp_mask =  ( Length = batch_size ) 

        embeddings = self.embedding_layer(inp) * math.sqrt(self.d_model)  # [batch_size, max_seq_length, embed_dim]
        pos_encoded_embeddings = self.pos_encoder(embeddings,inp_mask)
        src_attention_mask = self.create_src_mask(inp_mask)
        src_attention_mask = src_attention_mask.to(self.device_train)
        output = self.transformer_encoder(pos_encoded_embeddings,src_key_padding_mask=src_attention_mask)
        cls_vector = output[:,0,:]
        out = F.relu(self.fc_dropout(self.fc1(cls_vector)))
        out = F.relu(self.fc_dropout(self.fc2(out)))
        out = self.sigmoid(self.fc3(out))
        return out


# In[ ]:


t = TransformerEncoderModel(nlayers=4)
t  = t.to(device_fast)
inp = batch_data['input'].to(device_fast)
#inp = torch.randint(0,30522,(2,7))
#inp_mask = [4,7]
out = t(inp,batch_data['lengths'])


# In[13]:


import os
from torch.utils.tensorboard import SummaryWriter
from datetime import  datetime

def train(
        model,
        train_dataloader,valid_dataloader,
        num_epochs,
        criterion,optimizer,
        check_point_name,tensorboard_name,
        device_train = device_fast,use_rnn = False,log=True
    ):
    
    if log == True:
        current_datetime = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        tensorboard_name = tensorboard_name + "_" + current_datetime
        writer = SummaryWriter('runs/' + tensorboard_name)
    


    model = model.to(device_train)

    

    best_validation_loss = 1000.0
    valdiation_loss_not_decreased_steps = 0
    
    model.train()
    for e in range(num_epochs):
        
        training_set_size = 0
        training_loss = 0.0
        model.train()

        for data in tqdm(train_dataloader):
            
            optimizer.zero_grad()
            #input_reviews,inp_lengths,output_labels = data['input'], data['lengths'],data['labels']
            input_reviews, input_lengths , attention_mask ,output_labels = data['input'], data['lengths'], data['attention_mask'], data['labels']
            
            
            #inp = tokenizer(input_review,padding=True,truncation=True,return_tensors='pt')

            #input_reviews = inp['input_ids']
            #attention_mask = inp['attention_mask']

            input_reviews = input_reviews.to(device_train)
            attention_mask = attention_mask.to(device_train)

            training_set_size += input_reviews.shape[0]
            
            #output = model(input_reviews,inp_lengths)

            if USE_PRETRAINED_MODEL:
                output = model(input_reviews,attention_mask).squeeze(1)
            else:
                output = model(input_reviews,input_lengths).squeeze(1)

            output = output.to(device_cpu)
            loss = criterion(output,output_labels.float())
            training_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        current_training_loss = training_loss / training_set_size
        if log==True:
            print("Epoch " + str(e) + " Average Training Loss = " +  str(current_training_loss))
            writer.add_scalars('Loss vs Epoch',{'train' : current_training_loss},e)
        
        model.eval()
        
        if valid_dataloader is None:
            continue
        
        validation_set_size  = 0 
        if e% VALIDATION_LOSS_COMPUTE_STEP==0:
            correct_count = 0
            validation_loss = 0

            for i,data in enumerate(valid_dataloader,0):
                #input_reviews,inp_lengths,output_labels = data['input'], data['lengths'],data['labels']
                
                #input_review, output_labels = data['input'] , data['labels']
                input_reviews, input_lengths , attention_mask ,output_labels = data['input'], data['lengths'], data['attention_mask'], data['labels']

                #inp = tokenizer(input_review,padding=True,truncation=True,return_tensors='pt')


                #input_reviews = inp['input_ids']
                #attention_mask = inp['attention_mask']

                input_reviews = input_reviews.to(device_train)
                attention_mask = attention_mask.to(device_train)

                #input_reviews = input_reviews.to(device_train)
                validation_set_size += input_reviews.shape[0]
                #output = model(input_reviews,attention_mask)
                if USE_PRETRAINED_MODEL:
                    output = model(input_reviews,attention_mask).squeeze(1)
                else:
                    output = model(input_reviews,input_lengths).squeeze(1)

                #output = model(input_reviews,inp_lengths)
                output = output.to(device_cpu)
                loss = criterion(output,output_labels.float())
                validation_loss += loss.item()
                nearest_class = torch.round(output)

                correct = (nearest_class == output_labels.float()).float()
                correct_count += correct.sum()
            correct_count = int(correct_count)
            current_validation_accuracy = (correct_count/validation_set_size)*100
            current_validation_loss = (1.0* validation_loss)/validation_set_size
            if log == True:
                print("Epoch " + str(e) + " " +  "Validation Loss = " + str(current_validation_loss) )
                print("Validation Set Accuracy = " + str((correct_count/validation_set_size)*100) )

                writer.add_scalar(' Validation Accuracy vs Epoch ',int((correct_count/validation_set_size)*100),e)
                writer.add_scalars('Loss vs Epoch',{'valid' : current_validation_loss},e)
            
            if log==True:
                if current_validation_loss < best_validation_loss:
                    valdiation_loss_not_decreased_steps = 0
                    torch.save(model.state_dict(),check_point_name)
                    best_validation_loss = current_validation_loss
                else:
                    valdiation_loss_not_decreased_steps +=1
         
        if log == True:
            if valdiation_loss_not_decreased_steps >= PATIENCE_PARAMETER:
                break


# In[14]:


EPOCHS = 100
model = PretrainedModel()
model.freeze_weights()
optimizer = optim.Adam(model.parameters(),lr=0.001)

train(model,train_dataloader,valid_dataloader,EPOCHS,nn.BCELoss(),optimizer,'checkpoints/pretrained_model_freezed.pth','PretrainedModel',device_fast)


# ### Test phase

# In[21]:


def test(model,test_data,sentence_lengths,attention_mask,test_labels,device_test):
   
    #model.load_state_dict(torch.load(model_name))
    model.eval()
    test_labels = torch.tensor(test_labels)
    test_labels = test_labels.to(device_test)

    count = 0
    
    for i in range(len(test_data)):
        model = model.to(device_test)
        data_point = test_data[i].to(device_test)
        mask = attention_mask[i].to(device_test)
        if USE_PRETRAINED_MODEL:
            ans = model(data_point,mask)
        else:
            ans = model(data_point,[sentence_lengths[i]])
        
        ans = torch.round(ans)
        if ans[0][0] == test_labels[i]:
            count+=1
    
    print("Accuracy = " + str((count/len(test_data)*100)))




# In[19]:


test_review_ids = [] 
test_attention_masks = []
test_sentence_lengths = []
test_dataset_labels = []  

def getAssignmentTestData(load_from_trained=True):
    test_processed_text = []
    if not load_from_trained:
        with open("./E0334 Assignment2 Test Dataset.csv",encoding='utf-8') as csvfile:
            csvFile = csv.reader(csvfile)
            json_writer = jsonlines.open('test.jsonl','w')
            next(csvFile)
            for line in csvFile:
                processed_text = preprocess_text(line[0])
                label = 1.0 if line[1] == 'positive' else 0.0
                json_writer.write({"text":processed_text,"label":label})
                test_dataset_labels.append(label)
                test_processed_text.append(processed_text)
            json_writer.close()
    else:
        with open('./test.jsonl' ,encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                test_dataset_labels.append(sample['label'])
                test_processed_text.append(sample['text'])
      
    
    
    
    for i in range(len(test_processed_text)):

        if USE_PRETRAINED_MODEL:
            ids = berttokenizer(test_processed_text[i],padding=True,truncation=True,return_tensors="pt")
        else:
            ids = berttokenizer(test_processed_text[i],padding=True,truncation=False,return_tensors="pt")
        
        test_review_ids.append(ids['input_ids'])
        lengths = torch.sum(ids['attention_mask'],dim=1).tolist()
        test_sentence_lengths.append(lengths[0])
        test_attention_masks.append(ids['attention_mask'])
    
        #current_embeddings,current_sent_lengths,current_n_sent = review_to_embed(test_processed_text[i]) 
        #test_word_embeddings.append(current_embeddings.clone().detach().unsqueeze(0))
        #test_n_sents.append(current_n_sent)
        #test_sentence_lengths.append([current_sent_lengths])

getAssignmentTestData(load_from_trained=True)


# In[22]:


model = PretrainedModel()
model.load_state_dict(torch.load('checkpoints/pretrained_model_freezed.pth'))

test(model,test_review_ids,test_sentence_lengths,test_attention_masks,test_dataset_labels,device_fast)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




