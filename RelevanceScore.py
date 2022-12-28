
from transformers import BertTokenizer, BertForMaskedLM
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained("UBC-NLP/MARBERT",local_files_only=False)
model = BertForMaskedLM.from_pretrained("UBC-NLP/MARBERT",local_files_only=False)

def getLoss(true_sentence ,masked_sentence):
    inputs = tokenizer(masked_sentence, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    # retrieve index of [MASK]
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    # print(logits.shape)
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    # print(predicted_token_id)
    labels = tokenizer(true_sentence, return_tensors="pt")["input_ids"]
    # mask labels of non-[MASK] tokens
    labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

    outputs = model(**inputs, labels=labels)
    return (tokenizer.decode(predicted_token_id), round(outputs.loss.item(), 2))

def getTopPred(masked_sentence,topN = 100):
    inputs = tokenizer(masked_sentence, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index]    
    topN_tokend_id = np.argpartition(predicted_token_id.reshape(-1), -topN)[-topN:]
    return tokenizer.decode(topN_tokend_id)


import random
def getRelevanceScore(true_sentence, word):
    
    masked_sentence = true_sentence.replace(word,"[MASK]")
    pred_top_100 = getTopPred(masked_sentence)
    irrelevant_word = random.choices(list(tokenizer.vocab.keys()- pred_top_100.split()), k=1)[0]    
    (predicted, loss_actual) = getLoss(true_sentence, masked_sentence)
    
    true_sentence2 = true_sentence
    for pred in predicted.split(" "):
        true_sentence2 = true_sentence2.replace(word, pred,1)
        
    (predicted, loss_relevant) = getLoss(true_sentence2, masked_sentence)
    n_tokens = len(tokenizer.tokenize(irrelevant_word))
    (predicted, loss_not_relevant) = getLoss(true_sentence.replace(word, irrelevant_word), masked_sentence = true_sentence.replace(word, " ".join(["[MASK]"]*n_tokens)))
    point = .5 * loss_actual 
    if word not in pred_top_100:
        point = -1* point
        
    loss_actual = loss_actual- point
    loss_not_relevant = loss_not_relevant + point
    
    actual_relevant_dist = abs( loss_actual - loss_relevant  )
    actual_not_relevant_dist = abs( loss_actual - loss_not_relevant )

    total_distance  = actual_relevant_dist + actual_not_relevant_dist

    actual_relevant_precent = actual_not_relevant_dist  / total_distance
    actual_notrelevant_precent = actual_relevant_dist / total_distance

    pred = 0 if actual_relevant_precent >= .5 else 1
    return pred, actual_relevant_precent, actual_notrelevant_precent, predicted, irrelevant_word
