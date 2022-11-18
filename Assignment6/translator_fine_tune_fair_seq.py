#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
from torch.utils.data import DataLoader,Dataset
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from transformers import DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split
import torch
import evaluate

NUM_EPOCHS = 1
BATCH_SIZE = 8
metric = evaluate.load("sacrebleu")
#prompt = "translate English to German: "
mname = "facebook/wmt19-en-de"


# In[14]:


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
#translation_model = FSMTForConditionalGeneration.from_pretrained(mname,cache_dir='./.cache')
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


# In[17]:


train_dataset = TranslationDataset(train_df)
valid_dataset = TranslationDataset(valid_df)


# In[ ]:


from transformers import Seq2SeqTrainer,Seq2SeqTrainingArguments


model_name = "en_de_translator"

args = Seq2SeqTrainingArguments(
    f"MT5_EN_DE",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=NUM_EPOCHS,
    predict_with_generate=True,
    push_to_hub=True,
    evaluation_strategy = "epoch",
    save_strategy="steps",
    save_steps = 10000,
)



# In[18]:


import numpy as np

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = t5_tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != 1, labels, t5_tokenizer.pad_token_id)
    decoded_labels = t5_tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != t5_tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# In[11]:


trainer = Seq2SeqTrainer(
    translation_model,
    args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    tokenizer=t5_tokenizer,
    compute_metrics=compute_metrics,
)


# In[ ]:


trainer.train()


# In[ ]:


trainer.push_to_hub()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


'''
BATCH_SIZE = 3

train_dataloader = DataLoader(train_dataset,BATCH_SIZE,collate_fn=collate_fn,shuffle=True)
valid_dataloader = DataLoader(valid_dataset,BATCH_SIZE,collate_fn=collate_fn,shuffle=False)


# In[ ]:


batch_data = next(iter(train_dataloader))
print(batch_data)

outputs = translation_model(**batch_data)
print(outputs)
'''
'''
logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)


compute_metrics((predictions,batch_data['labels']))

'''


# In[ ]:


'''from torch.optim import AdamW
optimizer = AdamW(translation_model.parameters(), lr=5e-5)
from transformers import get_scheduler

NUM_EPOCHS = 4
num_training_steps = NUM_EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)'''


# In[ ]:


'''from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))

translation_model.train()

for epoch in range(NUM_EPOCHS):
    for batch in train_dataloader:
        outputs = translation_model(batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)'''


# In[ ]:


'''import evaluate

metric = evaluate.load("accuracy")
translation_model.eval()

for batch in valid_dataloader:
    with torch.no_grad():
        outputs = translation_model(**batch)

    logits = outputs.logits'''

