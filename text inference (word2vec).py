import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_word2vec_embedding(word):
    if word in model:
        return model[word]
    else:
        print(f"{word} not in vocabulary")
        return None
    

if __name__ == '__main__':
    model = api.load('word2vec-google-news-300') 

    abrasion = get_word2vec_embedding("abrasion")
    burn = get_word2vec_embedding("burn")
    bruises = get_word2vec_embedding("bruises")
    cut = get_word2vec_embedding("cut")
    ingrown_nail = get_word2vec_embedding("ingrown_nail")
    laceration = get_word2vec_embedding("laceration")
    stab_wound = get_word2vec_embedding("stab_wound")

    look_up = [abrasion, burn, bruises, cut, ingrown_nail, laceration, stab_wound]
    look_up_word = ['abrasion', 'burn', 'bruises', 'cut', 'ingrown_nail', 'laceration', 'stab_wound']


    sentence = "Today I went to the kitchen, the floor is slippery, I accidentally fell down and injured myself"
    threshold = 0.1


    sum_scores = [0.0] * len(look_up)
    words = sentence.lower().replace(",", "").split()

    
    for word in words:
        vector = get_word2vec_embedding(word)
        if vector is not None:
            irrelevant = True
            value_list = []
            for i in range(len(sum_scores)):

                if look_up[i] is not None:  
                    value = cosine_similarity([vector], [look_up[i]])[0][0] 
                    if abs(value) > 0.1:
                        irrelevant = False 
                    value_list.append(value)
                    print(f"{look_up_word[i]} and {word} similarity: {value}")
            if (not irrelevant):
                for i in range (len(sum_scores)):
                    sum_scores[i] += value_list[i]

    print("Final accumulated similarity scores:", sum_scores)
