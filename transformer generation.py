
#Generation of text using GPT2 model
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

seed_text = "Once upon a time"
input_ids = tokenizer.encode(seed_text, return_tensors='pt')

output = model.generate(input_ids, max_length=40, temperature=0.7,
    no_repeat_ngram_size=2,
    pad_token_id=tokenizer.eos_token_id)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)


#Translation with T5 model
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_prompt = "translate English to French: 'Hello, how are you?'"
input_ids = tokenizer.encode(input_prompt, return_tensors="pt")

output = model.generate(input_ids, max_length=100)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:", generated_text)

#evaluation with BLEU score
from torchmetrics.text import BLEUScore

generated_text = ['the cat is on the mat']
real_text = [['there is a cat on the mat', 'a cat is on the mat']]

bleu = BLEUScore()
bleu_metric = bleu(generated_text, real_text)

print("BLEU Score: ", bleu_metric.item())

#evaluation with ROUGE score
from torchmetrics.text import ROUGEScore

generated_text = 'Hello, how are you doing?'
real_text = "Hello, how are you?"

rouge = ROUGEScore()
rouge_score = rouge([generated_text], [[real_text]])

print("ROUGE Score:", rouge_score)

