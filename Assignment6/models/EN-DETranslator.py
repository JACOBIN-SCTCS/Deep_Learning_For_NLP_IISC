from transformers import MT5ForConditionalGeneration, T5Tokenizer

model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")


