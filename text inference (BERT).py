from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
import re

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors='pt')
    with torch.no_grad(): 
        outputs = model(**inputs)
    return outputs.last_hidden_state[0][0]
    




def get_sentence_embedding(sentence):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    # Disable gradient calculations since we're doing inference
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the embedding of the [CLS] token (the first token)
    return outputs.last_hidden_state.mean(dim=1)  # CLS token is the first token in the sequence



if __name__ == '__main__':
    abrasion = get_word_embedding("abrasion")
    burn = get_word_embedding("burn")
    bruises = get_word_embedding("bruises")
    cut = get_word_embedding("cut")
    ingrown_nail = get_word_embedding("ingrown nail")
    laceration = get_word_embedding("laceration")
    stab_wound = get_word_embedding("stab wound")

    look_up = [abrasion, burn, bruises, cut, ingrown_nail, laceration, stab_wound]
    look_up_word = ['abrasion', 'burn', 'bruises', 'cut', 'ingrown_nail', 'laceration', 'stab_wound']
    sentence = "I burn myself"

    sum = [0.0]*7
    print(sum)
    for word in sentence.lower().replace(",", "").split(" "):
        vector = get_word_embedding(word)
        for i in range(len(sum)):
            value = F.cosine_similarity(vector.unsqueeze(0), look_up[i].unsqueeze(0)).item()
            sum[i] += value
            print(look_up_word[i], word, value)

    print(sum)

    line_vector = get_sentence_embedding(sentence)
    for i in range(len(sum)):
        value = F.cosine_similarity(line_vector.unsqueeze(0), look_up[i].unsqueeze(0)).item()
        sum[i] = value
    print(sum)

