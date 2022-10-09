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

# SENTENCE_SPLITTING_USED; whether to use the splitting of reviews into sentences.
EMBED_DIM = 300
HIDDEN_DIM = 256
ATTENTION_DIM = 256
NUM_FILTERS = 86

PATIENCE_PARAMETER = 7
VALIDATION_LOSS_COMPUTE_STEP = 1
EXPAND_CONTRACTIONS = True


device_cpu = torch.device('cpu')
device_fast = torch.device('cpu')



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
    new_words = []
    for word in text.split(" "):
        #if word == 'n\'t':
        #    if len(new_words) > 1:
        #        new_words[-1] = new_words[-1] + word
        #    else:
        #        new_words.append(word)
        if word == '\'s':
            pass
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


def getWordEmbeddingforText(text,glove=glove):
    length = 0
    words = []
    text = text.strip()
    for word in text.split(' '):
        w = word.strip()
        if w=='':
            continue
        length+=1
        word_embedding = glove[w]
        words.append(word_embedding)
    
    return torch.stack(words),length


# In[8]:


# Sentences, word
def review_to_embed(review,glove=glove): 
    sentences = review.split(".")
    sentence_lengths = []
    review_embeddings = []
    num_sentences = 0
    for sentence in sentences:
        s = sentence.strip()
        if s == '':
            continue
        num_sentences += 1
        sentence_word_embeddings,sentence_length = getWordEmbeddingforText(s,glove)
        sentence_lengths.append(sentence_length)
        review_embeddings.append(sentence_word_embeddings)

    return torch.nn.utils.rnn.pad_sequence(review_embeddings,batch_first=True),sentence_lengths,num_sentences


# In[9]:


class ReviewDataSet(Dataset):
    
    def __init__(self,reviews):
        super().__init__()
        self.reviews = reviews
        
    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        return self.reviews[index]


# In[10]:


processed_dataset = []
for review in preprocessed_dataset:
        embeddings, sent_length ,n_sents = review_to_embed(review['text'])
        processed_dataset.append({'review': embeddings,'sent_lengths': sent_length,'length' : n_sents,'label' : review['label']})


# In[11]:


def collate_function(batch_data):   
    
    inputs = [b['review'] for b in batch_data]
    sent_lengths = [ b['sent_lengths'] for b in batch_data ]
    n_sentences = [ b['length'] for b in batch_data ]
    labels = torch.tensor([b['label'] for b in batch_data])


    labels = labels.unsqueeze(1)
    max_n_sentences = max([i.shape[0] for i in inputs] )
    max_n_words = max([i.shape[1] for i in inputs])

 
    processed_inputs = []
    for inp in inputs:

        t1 = torch.permute(inp,(2,1,0))
        t1 = torch.nn.functional.pad(t1,(0,max_n_sentences-inp.shape[0],0,max_n_words-inp.shape[1]))
        t1 = torch.permute(t1,(2,1,0))
        processed_inputs.append(t1)

    final_inp = torch.stack(processed_inputs)
    #inputs = pad_sequence(inputs,batch_first=True)
    return  {'input' : final_inp , 'sent_lengths': sent_lengths , 'lengths' : n_sentences ,'labels' : labels }


# In[12]:


train_idx,valid_idx = train_test_split(np.arange(train_dataset_labels.shape[0]), 
    test_size=0.2,
    shuffle= True,
    stratify= train_dataset_labels,
    random_state=0
)

dataset = ReviewDataSet(processed_dataset)
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
train_dataloader = DataLoader(dataset,16,sampler=train_sampler,collate_fn=collate_function)
valid_dataloader = DataLoader(dataset,16,sampler=valid_sampler,collate_fn=collate_function)


# In[13]:


batch_data = next(iter(train_dataloader))


# ## HAN

# In[14]:


import torch.nn.functional as F

class WordAttention(nn.Module):

    def __init__(self,
        embed_dim=EMBED_DIM,
        hidden_dim = HIDDEN_DIM,
        attention_dim = ATTENTION_DIM,
        num_layers=1,
        bidirectional=True,
        device_train=device_cpu,
        rnn_dropout = 0.0,
        fc_dropout = 0.3,
    ):
        super().__init__()
        self.rnn = nn.GRU(embed_dim,hidden_dim,num_layers=num_layers,batch_first=True,bidirectional=bidirectional,dropout=rnn_dropout)
        bidirectional_factor = 2 if bidirectional else 1
        self.word_attention = nn.Linear(bidirectional_factor*hidden_dim,attention_dim)
        self.u_w = nn.Linear(attention_dim,1)
        self.device_train = device_train
        self.fc_dropout = nn.Dropout(fc_dropout)
    
    def create_mask(self,inp_len):

        mask = torch.ones(len(inp_len),max(inp_len),dtype=torch.int64)
        for i in range(len(inp_len)):
            mask[i,inp_len[i]:] = 0
        return mask
        
    def forward(self,inp,inp_len):
        
        # inp = 1 review  = [num_sentences , num_words , embed_dim]
        # inp_len = length = num_sentences , each element number of words in  sentence.

        packed_embedding = nn.utils.rnn.pack_padded_sequence(inp,inp_len,batch_first=True,enforce_sorted=False)
        packed_output,hidden = self.rnn(packed_embedding)
        outputs,_ = nn.utils.rnn.pad_packed_sequence(packed_output,batch_first=True)
        
        attention_outs = torch.tanh(self.fc_dropout(self.word_attention(outputs)))
        attention_scores = self.u_w(attention_outs)
        attention_scores = attention_scores.squeeze(2)
        attention_mask = self.create_mask(inp_len).to(self.device_train)
        attention_scores = attention_scores.masked_fill(attention_mask==0, -1e10)   # Fill padding tokens with a lower value
        attention_probs = F.softmax(attention_scores,dim=1)
        attention_probs = attention_scores.unsqueeze(2)

        weighted_embeddings = attention_probs * outputs
        output = torch.sum(weighted_embeddings,dim=1)
        return output


# In[15]:


class SentenceAttention(nn.Module):
    
    def __init__(self,
            embed_dim=EMBED_DIM,
            hidden_dim = HIDDEN_DIM,
            attention_dim=ATTENTION_DIM,
            num_layers=1,
            bidirectional=True,
            train_device = device_cpu,
            rnn_dropout = 0.0,
            fc_dropout = 0.3
        ):
        
        super().__init__()
        self.rnn = nn.GRU(embed_dim,hidden_dim,num_layers=num_layers,batch_first=True,bidirectional=bidirectional,dropout=rnn_dropout)
        bidirectional_factor = 2 if bidirectional else 1
        self.sentence_attention = nn.Linear(bidirectional_factor*hidden_dim,attention_dim)
        self.u_s = nn.Linear(attention_dim,1)
        self.train_device = train_device
        self.fc_dropout = nn.Dropout(fc_dropout)

    def create_mask(self,sent_len):
        mask = torch.ones(len(sent_len),max(sent_len),dtype=torch.int64)
        for i in range(len(sent_len)):
            mask[i,sent_len[i]:] = 0
        return mask
    
 
    def forward(self,sents,sent_len):
        
        packed_embedding = nn.utils.rnn.pack_padded_sequence(sents,sent_len,enforce_sorted=False)
        packed_output,hidden = self.rnn(packed_embedding)
        outputs,_ = nn.utils.rnn.pad_packed_sequence(packed_output,batch_first=True)

        attention_outs = torch.tanh(self.fc_dropout(self.sentence_attention(outputs)))
        attention_scores = self.u_s(attention_outs)
        attention_scores = attention_scores.squeeze(2)
        attention_mask = self.create_mask(sent_len).to(self.train_device)
        attention_scores = attention_scores.masked_fill(attention_mask==0, -1e10)   # Fill padding tokens with a lower value
        attention_probs = F.softmax(attention_scores,dim=1)
        attention_probs = attention_scores.unsqueeze(2)
        weighted_embeddings = attention_probs*outputs
        output = torch.sum(weighted_embeddings,dim=1)
        return output


# In[16]:


class HierarchialAttention(nn.Module):

    def __init__(self,
                
                input_embed_dim = EMBED_DIM,
                word_encoder_hidden_dim = HIDDEN_DIM,
                word_encoder_num_layers = 1,
                word_encoder_bidirectional = True,
                word_encoder_attention_dim = HIDDEN_DIM,
                word_encoder_fc_dropout = 0.3,

                sentence_encoder_hidden_dim = HIDDEN_DIM,
                sentence_encoder_num_layers = 1,
                sentence_encoder_bidirectional= True,
                sentence_encoder_attention_dim = HIDDEN_DIM,
                sentence_encoder_fc_dropout = 0.3,

                rnn_dropout = 0.0,
                fc_dropout = 0.3,
                train_device = device_cpu
            ):
      
        super().__init__()

        rnn_dropout_word = 0.3 if word_encoder_num_layers > 1 else  0.0
        rnn_dropout_sentence = 0.3 if sentence_encoder_num_layers > 1 else 0.0

        self.word_encoder = WordAttention(input_embed_dim,word_encoder_hidden_dim,word_encoder_attention_dim,word_encoder_num_layers,word_encoder_bidirectional,train_device,rnn_dropout_word,word_encoder_fc_dropout)
        bidirectional_factor = 2 if word_encoder_bidirectional else 1
        self.sentence_encoder = SentenceAttention(bidirectional_factor*word_encoder_hidden_dim,sentence_encoder_hidden_dim,sentence_encoder_attention_dim,sentence_encoder_num_layers,sentence_encoder_bidirectional,train_device,rnn_dropout_sentence,sentence_encoder_fc_dropout)
        
        division_factor = 2
        self.fc_list = [
                nn.Linear(bidirectional_factor*sentence_encoder_hidden_dim,sentence_encoder_hidden_dim),
                nn.Linear(sentence_encoder_hidden_dim,sentence_encoder_hidden_dim>>1),
        ] 

        #for i in range(division_factor):
        #    self.fc_list.append(nn.Linear(sentence_encoder_hidden_dim/(2**i),sentence_encoder_hidden_dim/(2**(i+1))))
        
        self.fc = nn.ModuleList(self.fc_list)
        #self.fc_out = nn.Linear(sentence_encoder_hidden_dim/(2**(division_factor)),1)
        self.fc_out = nn.Linear(sentence_encoder_hidden_dim>>1,1)
        #self.fc_out = nn.Linear(sentence_encoder_hidden_dim,1)
        self.fc_dropout_layer = nn.Dropout(p=fc_dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self,inp,inp_sentence_lengths,inp_words_lengths):
        
        sentence_embeddings = []
        for i in range(inp.shape[0]):
            sentence_embeddings.append(self.word_encoder(inp[i],inp_words_lengths[i]))
        
        
        batch_sentences = pad_sequence(sentence_embeddings)
        #batch_sentences = torch.stack(sentence_embeddings)
        doc_embedding = self.sentence_encoder(batch_sentences,inp_sentence_lengths)

        out = doc_embedding
        for i,l in enumerate(self.fc_list):
            out = self.fc_dropout_layer(F.relu(l(out)))
        
        out = self.sigmoid(self.fc_out(out))
        return out


# ### Hierarchial Model with Self Attention

# In[17]:


import torch.nn.functional as F

class WordSelfAttention(nn.Module):

    def __init__(self,
        embed_dim=EMBED_DIM,
        hidden_dim = HIDDEN_DIM,
        attention_dim = ATTENTION_DIM,
        num_layers=1,
        bidirectional=True,
        device_train=device_cpu,
        rnn_dropout = 0.0,
        fc_dropout = 0.3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.rnn = nn.GRU(embed_dim,hidden_dim,num_layers=num_layers,batch_first=True,bidirectional=bidirectional,dropout=rnn_dropout)
        bidirectional_factor = 2 if bidirectional else 1
        
        self.query_attention = nn.Linear(bidirectional_factor*hidden_dim,attention_dim)
        self.key_attention = nn.Linear(bidirectional_factor*hidden_dim,attention_dim)
       
        self.device_train = device_train
        self.fc_dropout = nn.Dropout(fc_dropout)
    
    def create_mask(self,inp_len):

        mask = torch.ones(len(inp_len),max(inp_len),dtype=torch.int64)
        for i in range(len(inp_len)):
            mask[i,inp_len[i]:] = 0
        return mask
    
    def create_self_attention_mask(self,inp_len):
        mask = torch.ones(len(inp_len),max(inp_len),max(inp_len),dtype=torch.int64)
        for i in range(len(inp_len)):
            mask[i,:,inp_len[i]:] = 0 
            mask[i,inp_len[i]:,:] = 0
        return mask


    def forward(self,inp,inp_len):
        
        # inp = 1 review  = [num_sentences , num_words , embed_dim]
        # inp_len = length = num_sentences , each element number of words in  sentence.

        packed_embedding = nn.utils.rnn.pack_padded_sequence(inp,inp_len,batch_first=True,enforce_sorted=False)
        packed_output,hidden = self.rnn(packed_embedding)
        outputs,_ = nn.utils.rnn.pad_packed_sequence(packed_output,batch_first=True)
        
        queries = torch.tanh(self.query_attention(outputs))
        keys = torch.tanh(self.key_attention(outputs))
        
        P_matrix = torch.matmul(queries,torch.transpose(keys,1,2)) * (1/np.sqrt(self.embed_dim))

        attention_mask = self.create_self_attention_mask(inp_len).to(self.device_train)
        normalized_P_matrix = P_matrix.masked_fill(attention_mask==0,-1e10)
        attention_probs = F.softmax(normalized_P_matrix,dim=2)

        final_outputs = torch.matmul(attention_probs,outputs)
        output = torch.sum(final_outputs,dim=1)

        return output
        


# In[18]:


class SentenceSelfAttention(nn.Module):
    
    def __init__(self,
            embed_dim=EMBED_DIM,
            hidden_dim = HIDDEN_DIM,
            attention_dim=ATTENTION_DIM,
            num_layers=1,
            bidirectional=True,
            train_device = device_cpu,
            rnn_dropout = 0.0,
            fc_dropout = 0.3
        ):
        
        super().__init__()
        self.embed_dim = embed_dim
        self.rnn = nn.GRU(embed_dim,hidden_dim,num_layers=num_layers,batch_first=True,bidirectional=bidirectional,dropout=rnn_dropout)
        bidirectional_factor = 2 if bidirectional else 1
      

        self.query_attention = nn.Linear(bidirectional_factor*hidden_dim,attention_dim)
        self.key_attention = nn.Linear(bidirectional_factor*hidden_dim,attention_dim)
        self.device_train = train_device
        self.train_device = train_device
        self.fc_dropout = nn.Dropout(fc_dropout)

    def create_mask(self,sent_len):
        mask = torch.ones(len(sent_len),max(sent_len),dtype=torch.int64)
        for i in range(len(sent_len)):
            mask[i,sent_len[i]:] = 0
        return mask

    def create_self_attention_mask(self,sent_len):
        mask = torch.ones(len(sent_len),max(sent_len),max(sent_len),dtype=torch.int64)
        for i in range(len(sent_len)):
            mask[i,:,sent_len[i]:] = 0 
            mask[i,sent_len[i]:,:] = 0
        return mask


    def forward(self,sents,sent_len):
        
        packed_embedding = nn.utils.rnn.pack_padded_sequence(sents,sent_len,enforce_sorted=False)
        packed_output,hidden = self.rnn(packed_embedding)
        outputs,_ = nn.utils.rnn.pad_packed_sequence(packed_output,batch_first=True)
 
        queries = torch.tanh(self.query_attention(outputs))
        keys = torch.tanh(self.key_attention(outputs))
        
        P_matrix = torch.matmul(queries,torch.transpose(keys,1,2)) * (1/np.sqrt(self.embed_dim))

        attention_mask = self.create_self_attention_mask(sent_len).to(self.device_train)
        normalized_P_matrix = P_matrix.masked_fill(attention_mask==0,-1e10)
        attention_probs = F.softmax(normalized_P_matrix,dim=2)

        final_outputs = torch.matmul(attention_probs,outputs)
        output = torch.sum(final_outputs,dim=1)

        return output


# In[19]:


class HierarchialSelfAttention(nn.Module):

    def __init__(self,
                
                input_embed_dim = EMBED_DIM,
                word_encoder_hidden_dim = HIDDEN_DIM,
                word_encoder_num_layers = 1,
                word_encoder_bidirectional = True,
                word_encoder_attention_dim = HIDDEN_DIM,
                word_encoder_fc_dropout = 0.3,

                sentence_encoder_hidden_dim = HIDDEN_DIM,
                sentence_encoder_num_layers = 1,
                sentence_encoder_bidirectional= True,
                sentence_encoder_attention_dim = HIDDEN_DIM,
                sentence_encoder_fc_dropout = 0.3,

                rnn_dropout = 0.0,
                fc_dropout = 0.3,
                train_device = device_cpu
            ):
      
        super().__init__()

        rnn_dropout_word = 0.3 if word_encoder_num_layers > 1 else  0.0
        rnn_dropout_sentence = 0.3 if sentence_encoder_num_layers > 1 else 0.0

        self.word_encoder = WordSelfAttention(input_embed_dim,word_encoder_hidden_dim,word_encoder_attention_dim,word_encoder_num_layers,word_encoder_bidirectional,train_device,rnn_dropout_word,word_encoder_fc_dropout)
        bidirectional_factor = 2 if word_encoder_bidirectional else 1
        self.sentence_encoder = SentenceSelfAttention(bidirectional_factor*word_encoder_hidden_dim,sentence_encoder_hidden_dim,sentence_encoder_attention_dim,sentence_encoder_num_layers,sentence_encoder_bidirectional,train_device,rnn_dropout_sentence,sentence_encoder_fc_dropout)
        
        division_factor = 2
        self.fc_list = [
                nn.Linear(bidirectional_factor*sentence_encoder_hidden_dim,sentence_encoder_hidden_dim),
                nn.Linear(sentence_encoder_hidden_dim,sentence_encoder_hidden_dim>>1)
        ] 

        #for i in range(division_factor):
        #    self.fc_list.append(nn.Linear(sentence_encoder_hidden_dim/(2**i),sentence_encoder_hidden_dim/(2**(i+1))))
        
        self.fc = nn.ModuleList(self.fc_list)
        #self.fc_out = nn.Linear(sentence_encoder_hidden_dim/(2**(division_factor)),1)
        self.fc_out = nn.Linear(sentence_encoder_hidden_dim>>1,1)
        #self.fc_out = nn.Linear(sentence_encoder_hidden_dim,1)
        self.fc_dropout_layer = nn.Dropout(p=fc_dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self,inp,inp_sentence_lengths,inp_words_lengths):
        
        sentence_embeddings = []
        for i in range(inp.shape[0]):
            sentence_embeddings.append(self.word_encoder(inp[i],inp_words_lengths[i]))
        
        
        batch_sentences = pad_sequence(sentence_embeddings)
        #batch_sentences = torch.stack(sentence_embeddings)
        doc_embedding = self.sentence_encoder(batch_sentences,inp_sentence_lengths)

        out = doc_embedding
        for i,l in enumerate(self.fc_list):
            out = self.fc_dropout_layer(F.relu(l(out)))
        
        out = self.sigmoid(self.fc_out(out))
        return out


# In[20]:


h_self_model = HierarchialSelfAttention()
h_self_model(batch_data['input'],batch_data['lengths'],batch_data['sent_lengths'])


# In[21]:


h_model = HierarchialAttention()
h_model(batch_data['input'],batch_data['lengths'],batch_data['sent_lengths'])


# In[22]:


import os
from torch.utils.tensorboard import SummaryWriter
from datetime import  datetime

def train(model,train_dataloader,valid_dataloader,num_epochs,criterion,optimizer,
    checkpoint_name='best_model.pt',
    device_train = device_fast,use_rnn = False,log=True):

    tensorboard_name='Ensemble'
    if log == True:
        current_datetime = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        tensorboard_name = tensorboard_name + "_" + current_datetime
        writer = SummaryWriter('runs/' + tensorboard_name)
    
    
    model = model.to(device_train)
    clip = 0
    if use_rnn:
        clip = 5

    best_validation_loss = 1000.0
    best_validation_accuracy = 0.0
    valdiation_loss_not_decreased_steps = 0
    
    model.train()
    for e in range(num_epochs):
        
        training_set_size = 0
        training_loss = 0.0
        model.train()

        for data in tqdm(train_dataloader):
            
            optimizer.zero_grad()
            input_reviews,sent_lengths,n_sents,output_labels = data['input'], data['sent_lengths'],data['lengths'],data['labels']
            input_reviews = input_reviews.to(device_train)
            training_set_size += input_reviews.shape[0]
            output = model(input_reviews,n_sents,sent_lengths)
            output = output.to(device_cpu)
            loss = criterion(output,output_labels.float())
            training_loss += loss.item()
            loss.backward()
            if use_rnn:
                nn.utils.clip_grad_norm_(model.parameters(),clip)
            optimizer.step()
        
        current_training_loss = training_loss
        if log==True:
            print("Epoch " + str(e) + " Average Training Loss = " +  str(current_training_loss))
            writer.add_scalars(tensorboard_name + 'Training Loss vs Epoch',{'train' : current_training_loss},e)

        
        model.eval()
        
        if valid_dataloader is None:
            continue
        
        validation_set_size  = 0 
        if e% VALIDATION_LOSS_COMPUTE_STEP==0:
            correct_count = 0
            validation_loss = 0

            for i,data in enumerate(valid_dataloader,0):
                
                input_reviews,sent_lengths,n_sents,output_labels = data['input'], data['sent_lengths'],data['lengths'],data['labels']
                input_reviews = input_reviews.to(device_train)
                validation_set_size += input_reviews.shape[0]
                output = model(input_reviews,n_sents,sent_lengths)
                output = output.to(device_cpu)
                loss = criterion(output,output_labels.float())
                validation_loss += loss.item()
                nearest_class = torch.round(output)

                correct = (nearest_class == output_labels.float()).float()
                correct_count += correct.sum()
            correct_count = int(correct_count)
            current_validation_accuracy = (correct_count/validation_set_size)*100
            current_validation_loss = (1.0* validation_loss)
            if log == True:
                print("Epoch " + str(e) + " " +  "Validation Loss = " + str(current_validation_loss) )
                print("Validation Set Accuracy = " + str((correct_count/validation_set_size)*100) )
                writer.add_scalar(tensorboard_name + ' Validation Accuracy vs Epoch ',(correct_count/validation_set_size*100),e)
                writer.add_scalars(tensorboard_name + 'Validation Loss vs Epoch',{'valid' : current_validation_loss},e)

            
            if log==True:
                if current_validation_loss < best_validation_loss:
                    valdiation_loss_not_decreased_steps = 0
                    torch.save(model.state_dict(),checkpoint_name)
                    best_validation_loss = current_validation_loss
                else:
                    if current_validation_accuracy >= best_validation_accuracy:
                        best_validation_accuracy = current_validation_accuracy
                        torch.save(model.state_dict(),'ValAcc' + checkpoint_name)
                    
                    valdiation_loss_not_decreased_steps +=1
        if log == True:
            if valdiation_loss_not_decreased_steps >= PATIENCE_PARAMETER:
                break


# In[23]:


input_embed_dim = EMBED_DIM
word_encoder_hidden_dim = HIDDEN_DIM
word_encoder_num_layers = 2
word_encoder_bidirectional = True
word_encoder_attention_dim = HIDDEN_DIM
word_encoder_fc_dropout = 0.3

sentence_encoder_hidden_dim = HIDDEN_DIM
sentence_encoder_num_layers = 2
sentence_encoder_bidirectional= True
sentence_encoder_attention_dim = HIDDEN_DIM
sentence_encoder_fc_dropout = 0.3

rnn_dropout = 0.0
fc_dropout = 0.3
train_device = device_cpu

hierarchial_model_rnn_dropout = 0.0
hierarchial_model_fc_dropout = 0.3

hierarchial_model = HierarchialSelfAttention(
      
        input_embed_dim = input_embed_dim,
        word_encoder_hidden_dim = word_encoder_hidden_dim,
        word_encoder_num_layers = word_encoder_num_layers,
        word_encoder_bidirectional = word_encoder_bidirectional,
        word_encoder_attention_dim = word_encoder_attention_dim,
        word_encoder_fc_dropout= word_encoder_fc_dropout,

        sentence_encoder_hidden_dim = sentence_encoder_hidden_dim,
        sentence_encoder_num_layers = sentence_encoder_num_layers,
        sentence_encoder_bidirectional= sentence_encoder_bidirectional,
        sentence_encoder_attention_dim = sentence_encoder_hidden_dim,
        sentence_encoder_fc_dropout= sentence_encoder_fc_dropout,

        rnn_dropout = hierarchial_model_rnn_dropout,
        fc_dropout = hierarchial_model_fc_dropout,
        train_device = device_fast  
)



# In[24]:


checkpoint_name = "Hierarchial" + str(input_embed_dim) + "_" +  str(word_encoder_hidden_dim) + "_" + str(word_encoder_num_layers) + "_" + str(word_encoder_bidirectional) + "_" + str(word_encoder_attention_dim) + "_" + str(sentence_encoder_attention_dim) + "_"+ str(sentence_encoder_num_layers) + "_" + str(sentence_encoder_bidirectional) + "_"+ str(sentence_encoder_attention_dim) + "_" + str(hierarchial_model_rnn_dropout) + "_"+ str(hierarchial_model_fc_dropout)+"_"+str(word_encoder_fc_dropout)+"_" +str(sentence_encoder_fc_dropout) +".pth"
train(hierarchial_model,train_dataloader,valid_dataloader,50,nn.BCELoss(),optim.Adam(hierarchial_model.parameters(),lr=0.001),checkpoint_name,device_train=device_fast)


# In[ ]:





# In[27]:


def test(model_name,test_data,sentence_lengths,test_lengths,test_labels):
    model = HierarchialAttention(

        input_embed_dim = input_embed_dim,
        word_encoder_hidden_dim = word_encoder_hidden_dim,
        word_encoder_num_layers = word_encoder_num_layers,
        word_encoder_bidirectional = word_encoder_bidirectional,
        word_encoder_attention_dim = word_encoder_attention_dim,

        sentence_encoder_hidden_dim = sentence_encoder_hidden_dim,
        sentence_encoder_num_layers = sentence_encoder_num_layers,
        sentence_encoder_bidirectional= sentence_encoder_bidirectional,
        sentence_encoder_attention_dim = sentence_encoder_hidden_dim,
        rnn_dropout = hierarchial_model_rnn_dropout,
        fc_dropout = hierarchial_model_fc_dropout,
        train_device = device_cpu
    )
    model.load_state_dict(torch.load(model_name,map_location=device_cpu))
    model.eval()
    count = 0
    for i in range(len(test_data)):
        ans = model(test_data[i],[test_lengths[i]],sentence_lengths[i])
        ans = torch.round(ans)
        if ans[0][0] == test_labels[i]:
            count+=1
    
    print("Accuracy = " + str((count/len(test_data)*100)))


# In[20]:


test_word_embeddings = [] 
test_n_sents = []
test_sentence_lengths = []
test_dataset_labels = []  

def getAssignmentTestData():
    test_processed_text = []
    with open("./E0334 Assignment2 Test Dataset.csv",encoding='utf-8') as csvfile:
        csvFile = csv.reader(csvfile)
        next(csvFile)
        for line in csvFile:
            processed_text = preprocess_text(line[0])
            label = 1.0 if line[1] == 'positive' else 0.0
            test_dataset_labels.append(label)
            test_processed_text.append(processed_text)

    for i in range(len(test_processed_text)):
        current_embeddings,current_sent_lengths,current_n_sent = review_to_embed(test_processed_text[i]) 
        test_word_embeddings.append(current_embeddings.clone().detach().unsqueeze(0))
        test_n_sents.append(current_n_sent)
        test_sentence_lengths.append([current_sent_lengths])

getAssignmentTestData()


# In[28]:


test('./300_256_2_True_256_256_2_True_256_0.0_0.3.pth',test_word_embeddings,test_sentence_lengths,test_n_sents,test_dataset_labels)


# In[ ]:




