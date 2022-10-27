#!/usr/bin/env python
# coding: utf-8

# In[10]:


from transformers import T5Tokenizer, T5ForConditionalGeneration , BertModel , BertTokenizer
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import Dataset
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchtext.vocab import vocab
from torchmetrics import Accuracy

train_dataset_name = 'ArithOpsTrain.xlsx'
df = pd.read_excel(train_dataset_name)
df = df.drop('Table 1',axis=1)
df = df.rename(columns=df.iloc[0]).loc[1:]

device_cpu = torch.device('cpu')
device_fast = torch.device('cpu')


if torch.cuda.is_available():
    device_fast = torch.device('cuda')


counters = {"[PAD]":1,"<SOS>":2,"<EOS>" : 3 , "+" : 4, "-" :5 , "*" : 6 , "/" : 7 }
for i in range(10):
    counters["number"+str(i)] = i + 8

algebraic_symbols = []
for i in range(len(df)):
    row = df.iloc[i]['Equation']
    current_algebraic_symbol = [0 for i in range(5)]
    for sym in row.split(' '):
        if sym in ['+','-','*','/','%']:
            current_algebraic_symbol[counters[sym]-4]+=1
    algebraic_symbols.append(str(current_algebraic_symbol))

df['algebraic_symbols'] = algebraic_symbols


output_vocabulary = vocab(counters,)

train_df , valid_df = train_test_split(df,test_size=0.1,random_state=0,stratify=df['algebraic_symbols'])


# In[3]:


class T5Dataset(Dataset):
    def __init__(
        self,
        data  : pd.DataFrame,
        tokenizer : T5Tokenizer,
        text_max_token_length = 512,
        output_max_token_length = 128
    ):
        
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data 
        self.text_max_token_length = text_max_token_length
        self.output_max_token_length = output_max_token_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        data_row = self.data.iloc[index]

        input_text = data_row["Description"]
        input_question = data_row["Question"]

        in_text = input_text + " [SEP] " + input_question
        
        input_text_encoding = self.tokenizer(
            in_text,
            max_length=self.text_max_token_length,
            padding = "max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        
        output_text = data_row["Equation"]        
        output_text = "<SOS> " + output_text + " <EOS>"
        output_tokens = output_text.split()

        output_tokens_id_full = torch.zeros((self.output_max_token_length,),dtype=torch.int64)
        output_tokens_id = torch.tensor(output_vocabulary.forward(output_tokens),dtype=torch.int64)
        
        output_tokens_id_full[:len(output_tokens)] = output_tokens_id

        output_attention_mask = torch.zeros((self.output_max_token_length,))
        output_attention_mask[:len(output_tokens_id)] = 1
        
        output_text_encoding = self.tokenizer(
            output_text,
            max_length=self.output_max_token_length,
            padding = "max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )


        return dict(
            input_text = input_text,
            output_text = output_text,
            input_text_ids = input_text_encoding['input_ids'].flatten(),
            input_attention_mask = input_text_encoding['attention_mask'].flatten(),
            output_text_ids = output_text_encoding['input_ids'].flatten(),
            output_attention_mask = output_text_encoding['attention_mask'].flatten(),
            output_text_ids_custom_tokenizer = output_tokens_id_full,
            output_attention_mask_custom_tokenizer = output_attention_mask,
        )  


# In[11]:


t5_tokenizer = T5Tokenizer.from_pretrained("t5-small",model_max_length=512)
special_tokens_dict = {'additional_special_tokens' : ['[SEP]']}
num_added_tokens = t5_tokenizer.add_special_tokens(special_tokens_dict)

t5_model =T5ForConditionalGeneration.from_pretrained("t5-small")
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[12]:


#train_dataset = T5Dataset(train_df,t5_tokenizer)
#valid_dataset = T5Dataset(valid_df,t5_tokenizer)
train_dataset = T5Dataset(train_df,bert_tokenizer)
valid_dataset = T5Dataset(valid_df,bert_tokenizer)


train_dataloader = DataLoader(train_dataset,32,True)
valid_dataloader = DataLoader(valid_dataset,32,shuffle=True)


# In[13]:


#batch_data = next(iter(train_dataloader))


# In[14]:


#batch_data


# In[ ]:


def postfix_evaluation(batch_data,input_values):

    arith_symbols = set(['+','-','*','/','%'])
    output_values = []
    
    for i in range(len(batch_data)):
        flag = True
        current_input = batch_data[i].split(' ')
        current_input.reverse()
        input_value = input_values[i]

        stack = []
        for symbol in current_input:
            if symbol in arith_symbols:
                if len(stack)<2:
                    flag = False
                    break
                in1 = stack.pop(-1)
                in2 = stack.pop(-1)

                res = 0
                if symbol=='+':
                    res = in1+in2
                elif symbol=='-':
                    res = in1 - in2 
                elif symbol == '*':
                    res = in1 * in2
                elif symbol=='/':
                    res = in1/in2
                else:
                    res = in1 % in2
                stack.append(res)


            else:
                if "number" in symbol:
                    index = int(symbol[6])
                    stack.append(input_value[index])

        if flag==False or len(stack)!=1:
            output_values.append(0)
        else:
            output_values.append(stack.pop(-1))

    ans = torch.tensor(output_values)
    return ans

ans = postfix_evaluation(["+ - number0 number1 number2","+ / - number0 number2 number1 number3"],[[1,4,6],[5,6,7,8]])


# In[ ]:


import math

class PositionalEncoding(nn.Module):

    def __init__(self,dim_model,dropout_p,max_len) -> None:
        super().__init__()
        self.dropout =  nn.Dropout(dropout_p)

        pos_encoding = torch.zeros(max_len,dim_model)
        
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) 
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) 
        
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        #pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)

        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:

        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(1), :])


# In[16]:


class TransformerModel(nn.Module):

    def __init__(
        self,
        num_tokens_input,
        num_tokens_output,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout_p
    ):
        super().__init__()

        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model,
            dropout_p= dropout_p,
            max_len=5000
        )

        self.src_embedding = nn.Embedding.from_pretrained(bert_model.embeddings.word_embeddings.weight,freeze=False)
        #self.src_embedding = nn.Embedding(num_tokens_input,dim_model)
        self.trg_embedding = nn.Embedding(num_tokens_output,dim_model)

        self.dim_model = dim_model

        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout= dropout_p,
            batch_first=True
        )

        self.out = nn.Linear(self.dim_model,num_tokens_output)

    
    def forward(self, src, trg, src_padding_mask=None,target_mask=None, target_padding_mask=None):

        src = self.src_embedding(src) * math.sqrt(self.dim_model)
        target = self.trg_embedding(trg) * math.sqrt(self.dim_model)
        #print(target.shape)
        src = self.positional_encoder(src)
        target = self.positional_encoder(target)
        
        transformer_out = self.transformer(
            src=  src,tgt = target,tgt_mask=target_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=target_padding_mask
        )
        out = self.out(transformer_out)
        return out
    
        
    def get_tgt_mask(self,size):
        
        mask = torch.tril(torch.ones(size,size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask==0,float('-inf'))
        mask = mask.masked_fill(mask==1,float(0.0))
        mask = mask.to(device_fast)
        return mask

    def get_padding_mask(self,matrix,pad_token):
        return (matrix==pad_token)


# In[ ]:


import torch.optim as optim
class TransformerTranslator(pl.LightningModule):

    def __init__(
        self,
        num_tokens_input,
        num_tokens_output,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout_p
    ):
        
        super().__init__()
        self.transformer = TransformerModel(
                num_tokens_input=num_tokens_input,
                num_tokens_output=num_tokens_output,
                dim_model=dim_model,
                num_heads=num_heads,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward= dim_feedforward,
                dropout_p=dropout_p
            )
        
        self.exactmatch_accuracy = Accuracy(mdmc_reduce='samplewise')


        self.loss_fn = nn.CrossEntropyLoss()


        
    def forward(self, src, trg, src_padding_mask=None,target_mask=None, target_padding_mask=None):

        return self.transformer(src,trg,src_padding_mask,target_mask,target_padding_mask)
        

    def training_step(self, batch_data,batch_idx):

        input_text_ids = batch_data['input_text_ids']
        input_attention_mask = batch_data['input_attention_mask']
        #output_text_ids = batch_data['output_text_ids']
        #output_attention_mask = batch_data['output_attention_mask']

        output_text_ids = batch_data['output_text_ids_custom_tokenizer']
        output_attention_mask = batch_data['output_attention_mask_custom_tokenizer']
        
        output_in = output_text_ids[:,:-1]
        output_expected = output_text_ids[:,1:]

        
        target_mask = self.transformer.get_tgt_mask(output_expected.shape[1])

        src_padding_mask = self.transformer.get_padding_mask(input_attention_mask,0)
        
        tgt_padding_mask = self.transformer.get_padding_mask(output_attention_mask[:,:-1],0)


        predictions = self(input_text_ids,output_in,src_padding_mask,target_mask,tgt_padding_mask)

        loss_value = None
        
        for i in range(predictions.shape[0]):
            if loss_value == None:
                loss_value = self.loss_fn(predictions[i],output_expected[i])
            else:
                loss_value += self.loss_fn(predictions[i],output_expected[i])

        train_loss = loss_value*(1.0/predictions.shape[0])


        #train_loss = self.loss_fn(predictions,output_expected)
        
        self.log("train_loss" , train_loss, prog_bar=True,logger=True)
  
        return train_loss

    def validation_step(self, batch_data,batch_idx):
        
        input_text_ids = batch_data['input_text_ids']
        input_attention_mask = batch_data['input_attention_mask']
        #output_text_ids = batch_data['output_text_ids']
        #output_attention_mask = batch_data['output_attention_mask']

        output_text_ids = batch_data['output_text_ids_custom_tokenizer']
        output_attention_mask = batch_data['output_attention_mask_custom_tokenizer']
        
        output_in = output_text_ids[:,:-1]
        output_expected = output_text_ids[:,1:]

        
        target_mask = self.transformer.get_tgt_mask(output_expected.shape[1])

        src_padding_mask = self.transformer.get_padding_mask(input_attention_mask,0)
        
        tgt_padding_mask = self.transformer.get_padding_mask(output_attention_mask[:,:-1],0)


        predictions = self(input_text_ids,output_in,src_padding_mask,target_mask,tgt_padding_mask)


        loss_value = None
        
        for i in range(predictions.shape[0]):
            if loss_value == None:
                loss_value = self.loss_fn(predictions[i],output_expected[i])
            else:
                loss_value += self.loss_fn(predictions[i],output_expected[i])

        valid_loss = loss_value*(1.0/predictions.shape[0])
        #valid_loss = self.loss_fn(predictions,output_expected)
        
        correct_predictions = torch.topk(predictions,1,dim=2).indices.squeeze(2)
        exact_match_acc = self.exactmatch_accuracy(correct_predictions,output_expected)

        metrics = {"valid_loss" : valid_loss, "valid_acc" : exact_match_acc}
        self.log_dict(metrics)
        return metrics

        #self.log("valid_loss" , valid_loss, prog_bar=True,logger=True)

        #return valid_loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(),lr = 0.0001)


# In[ ]:


class T5ArithTranslator(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")


    def forward(self, input_ids, input_attention_mask, decoder_attention_mask, labels):

        outs = self.t5_model(input_ids=input_ids,attention_mask = input_attention_mask,labels = labels)        
        return outs.loss ,  outs.logits

        
    def training_step(self, batch, batch_idx) :
        
        input_text_ids = batch["input_text_ids"]
        input_attention_mask = batch["input_attention_mask"]
        output_text_ids = batch["output_text_ids"]
        output_attention_mask = batch["output_attention_mask"]

        loss, outs = self(
            input_text_ids,
            input_attention_mask,
            output_attention_mask,
            output_text_ids
        )

        self.log("train_loss" , loss, prog_bar=True,logger=True)
        return loss 
    
    def validation_step(self, batch, batch_idx):
        
        input_text_ids = batch["input_text_ids"]
        input_attention_mask = batch["input_attention_mask"]
        output_text_ids = batch["output_text_ids"]
        output_attention_mask = batch["output_attention_mask"]

        loss, outs = self(
            input_text_ids,
            input_attention_mask,
            output_attention_mask,
            output_text_ids
        )

        self.log("valid_loss" , loss, prog_bar=True,logger=True)
        return loss 

    def configure_optimizers(self):
        return optim.Adam(self.parameters(),lr = 0.0001)


# In[ ]:


N_EPOCHS = 300
BATCH_SIZE = 32


checkpoint_callback = ModelCheckpoint(
    dirpath = "checkpoints",
    filename="transformer-scratch-adam-6l-16h-best-checkpoint",
    save_top_k = 1,
    verbose = True,
    monitor="valid_loss",
    mode = "min"
)

logger = TensorBoardLogger("transformer_scratch_adam_6l_16h_logs",name="transformertranslator")

trainer = pl.Trainer(
    logger = logger,
    callbacks =  checkpoint_callback,
    max_epochs=N_EPOCHS,
    log_every_n_steps=5,
    gpus=1,
    accelerator='gpu'
)


# In[ ]:


#model = T5ArithTranslator()
Num_tokens_input=30522
Num_tokens_output=len(output_vocabulary)
Dim_model=768
Num_heads=16
Num_encoder_layers=6
Num_decoder_layers=6
Dim_feedforward= 2048
Dropout_p=0.1

model = TransformerTranslator(
    Num_tokens_input,
    Num_tokens_output,
    Dim_model,
    Num_heads,
    Num_encoder_layers,
    Num_decoder_layers,
    Dim_feedforward,
    Dropout_p
)


# In[ ]:


trainer.fit(model,train_dataloader,valid_dataloader)


# ### Inference Model

# In[ ]:


#test_model = T5ArithTranslator.load_from_checkpoint(
#    '/Users/depressedcoder/DLNLP/Assignment5/partb/checkpoints/best-checkpoint-v1.ckpt'
#)
'''
test_model =  TransformerTranslator.load_from_checkpoint(
    '/Users/depressedcoder/DLNLP/Assignment5/partb/checkpoints/best-checkpoint-v1.ckpt'
)
test_model.freeze()

t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
special_tokens_dict = {'additional_special_tokens' : ['[SEP]']}
num_added_tokens = t5_tokenizer.add_special_tokens(special_tokens_dict)

bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[ ]:


def predict(model, input_sequence, max_length=128, SOS_token=1, EOS_token=2):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    model.eval()
    
    y_input = torch.tensor([[1]], dtype=torch.long, device=device_fast)

    num_tokens = len(input_sequence[0])

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.transformer.get_tgt_mask(y_input.size(1)).to(device_fast)
        
        pred = model(input_sequence, y_input, target_mask=tgt_mask)
        
        next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
        next_item = torch.tensor([[next_item]], device=device_fast)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS_token:
            break

    return y_input.view(-1).tolist()


# In[ ]:


test_input_ids = bert_tokenizer("last stop in their field trip was the aquarium . penny identified number0 species of sharks number1 species of eels and number2 different species of whales . [SEP] how many species was penny able to identify ?",return_tensors='pt').input_ids


# In[ ]:


predict(test_model,test_input_ids)


# In[ ]:


outputs = test_model.t5_model.generate(test_input_ids)


# In[ ]:


text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text.split(' '))


# In[ ]:

'''


