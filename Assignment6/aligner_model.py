# In[1]

from transformers import FSMTForConditionalGeneration,FSMTTokenizer,DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader


mname = "facebook/wmt19-en-de"
BATCH_SIZE = 3
device_cpu = torch.device('cpu')
device_fast = torch.device('cuda') if torch.cuda.is_available() else device_cpu

# In[]
    
class TranslationDataset(Dataset):

    def __init__(self,data_frame,tokenizer_name=mname) -> None:
        super().__init__()
        self.dataframe = data_frame
        self.tokenizer = FSMTTokenizer.from_pretrained(tokenizer_name,cache_dir='./.cache')
    
    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, index):
        tokenized_data = self.tokenizer(self.dataframe.iloc[index,0],text_target=self.dataframe.iloc[index,1], return_tensors="pt")
     
        tokenized_data['input_ids'] = tokenized_data['input_ids'].squeeze(0)
        tokenized_data['attention_mask'] = tokenized_data['attention_mask'].squeeze(0)
        tokenized_data['labels'] = tokenized_data['labels'].squeeze(0)
        return tokenized_data


# In[15]:


tokenizer_name = mname
translation_model = FSMTForConditionalGeneration.from_pretrained(mname)
t5_tokenizer = FSMTTokenizer.from_pretrained(mname,cache_dir='./.cache')

data_collator = DataCollatorForSeq2Seq(t5_tokenizer, model=translation_model, return_tensors="pt")
def collate_fn(batch_data):
    return data_collator(batch_data)


# In[16]:


data_frame = pd.read_csv("./data/EN-DE.txt", sep='\t',header=0, names=['src', 'trg', 'c1','c2','c3','c4','c5','c6'])[:100]
train_df , valid_df = train_test_split(data_frame,test_size=0.07,random_state=0)
train_df.to_csv('train_df.tsv',sep="\t")
valid_df.to_csv('valid_df.tsv',sep="\t")


train_dataset = TranslationDataset(train_df)
valid_dataset = TranslationDataset(valid_df)

train_dataloader = DataLoader(train_dataset,BATCH_SIZE,collate_fn=collate_fn,shuffle=True)
valid_dataloader = DataLoader(valid_dataset,BATCH_SIZE,collate_fn=collate_fn,shuffle=False)
# In[17]:

class TranslationAligner(nn.Module):

    def __init__(self,mha_dropout = 0.2,fc_dropout = 0.2):
        
        super().__init__()
        self.translation_model = FSMTForConditionalGeneration.from_pretrained('facebook/wmt19-en-de')
        self.multiheaded_attention = nn.MultiheadAttention(1024,8,dropout=0.2,batch_first=True)
        self.output_layer  = nn.Linear(1024,42024,dropout=0.2)
        
    def forward(self,batch_data):
        
        model_output = self.translation_model(**batch_data)
        attention_output = self.multiheaded_attention(model_output['decoder_hidden_states'][-1])
        output_score = self.output_layer(attention_output)
        return output_score
        
# In[]

import os
from torch.utils.tensorboard import SummaryWriter
from datetime import  datetime
import tqdm

PATIENCE_PARAMETER = 8

def train(model,train_dataloader,valid_dataloader,num_epochs,criterion,optimizer,check_point_name,tensorboard_name,device_train = device_fast,log=True):
    
    if log == True:
        current_datetime = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
        tensorboard_name = tensorboard_name + "_" + current_datetime
        writer = SummaryWriter('runs/' + tensorboard_name)
    
    model = model.to(device_train)
    
    best_validation_loss = 1000.0
    valdiation_loss_not_decreased_steps = 0
    gradient_accumulation_steps = 2

    
    model.train()
    for e in range(num_epochs):
        
        training_set_size = 0
        training_loss = 0.0
        model.train()

        for data in tqdm(train_dataloader):
            
            optimizer.zero_grad()
            
            data['input_ids'] = data['input_ids'].to(device_train)
            data['attention_mask'] = data['attention_mask'].to(device_train)
            data['labels'] = data['labels'].to(device_train)
            
            training_set_size += data['input_ids'].shape[0]

            output = model(data)

            output = output.to(device_cpu)

            loss = criterion(output,output_labels.float())
            loss = (loss / gradient_accumulation_steps)
            training_loss += loss.item()
    
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

        
        current_training_loss = training_loss / training_set_size
        print("Epoch " + str(e) + " Average Training Loss = " +  str(current_training_loss))
    
        if log==True:
            writer.add_scalars(tensorboard_name + '; Loss vs Epoch',{'train' : current_training_loss},e)

             
        model.eval()
        if valid_dataloader is None:
            continue

        validation_set_size  = 0 
        correct_count = 0
        validation_loss = 0

        for i,data in enumerate(valid_dataloader,0):
            
            data['input_ids'] = data['input_ids'].to(device_train)
            data['attention_mask'] = data['attention_mask'].to(device_train)
            data['labels'] = data['labels'].to(device_train)
            
            
            validation_set_size += data['input_ids'].shape[0]
            
            output = model(data)
            output = output.to(device_cpu)
            loss = criterion(output,output_labels.float())
            validation_loss += loss.item()
            nearest_class = torch.round(output)

            correct = (nearest_class == output_labels.float()).float()
            correct_count += correct.sum()
        
            torch.cuda.empty_cache()
        
            
        correct_count = int(correct_count)
        current_validation_accuracy = (correct_count/validation_set_size)*100
        current_validation_loss = (1.0* validation_loss)/validation_set_size
        print("Epoch " + str(e) + " " +  "Validation Loss = " + str(current_validation_loss) )
        print("Validation Set Accuracy = " + str((correct_count/validation_set_size)*100) )
        
        if current_validation_loss < best_validation_loss:
            valdiation_loss_not_decreased_steps = 0
            torch.save(model.state_dict(),check_point_name)
            best_validation_loss = current_validation_loss
        else:
            valdiation_loss_not_decreased_steps +=1
           
        if log == True:
            writer.add_scalar(tensorboard_name + ' Validation Accuracy vs Epoch ',current_validation_accuracy,e)
            writer.add_scalars('Loss vs Epoch',{'valid' : current_validation_loss},e)
            if valdiation_loss_not_decreased_steps >= PATIENCE_PARAMETER:
                break
        
# In[]

NUM_EPOCHS = 1         
translator_alignmentmodel = TranslationAligner()



# In[]        

from transformers import FSMTForConditionalGeneration, FSMTTokenizer
mname = "facebook/wmt19-en-de"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname)

input = "Machine learning is great, isn't it?"
input_ids = tokenizer.encode(input, return_tensors="pt")
outputs = model(input_ids,output_hidden_states=True)

# In[]
model
# %%
outputs['decoder_hidden_states'][-1].shape
# %%
outputs
# %%
