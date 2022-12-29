
from transformers import BertTokenizer, BertForMaskedLM
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained("UBC-NLP/MARBERT",local_files_only=False)
model = BertForMaskedLM.from_pretrained("UBC-NLP/MARBERT",local_files_only=False)

def getLoss(true_sentence ,masked_sentence):
    #  tokenize the input into ides and add the suitable tags
    inputs = tokenizer(masked_sentence, return_tensors="pt")

    # logits returns the output of the mode with shape (1, number of tokens, 100000 prediction for each token), ex. torch.Size([1, 9, 100000])
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # retrieve array of indexes of [MASK] in the sentence, ex. tensor([1, 5])
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    # get the maximum prediction value for each mask, ex. tensor([1932, 2079])
    predicted_token_id = logits[0, mask_token_index].argmax(axis=1)

    # get only the tokenized input ides
    labels = tokenizer(true_sentence, return_tensors="pt")["input_ids"]

    # mask labels of non-[MASK] tokens -100 for the cross intropy loss to ignore the evaluation
    labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

    outputs = model(**inputs, labels=labels)

    # seperate each predicted mask word in an element in the array accending order 
    # predicted_words = [tokenizer.decode(id) for id in predicted_token_id]
    return (tokenizer.decode(predicted_token_id), round(outputs.loss.item(), 2))

def getTopPred(masked_sentence,topN = 100):
    inputs = tokenizer(masked_sentence, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index]    
    topN_tokend_id = np.argpartition(predicted_token_id.reshape(-1), -topN)[-topN:]
    return tokenizer.decode(topN_tokend_id)

# def getTopPred(masked_sentence,topN = 100):
#     inputs = tokenizer(masked_sentence, return_tensors="pt")
#     with torch.no_grad():
#         logits = model(**inputs).logits
#     mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

#     # get top predicted for each mask
#     top_pred = [tokenizer.decode(np.argpartition(predicted_token_id.reshape(-1), -topN)[-topN:]) for predicted_token_id in logits[0, mask_token_index] ]  
#     return top_pred # shape of (number of masks, topN)


import random
import re 
def getRelevanceScore(true_sentence, word):
    report = []

    # indices of the word in case of many of the same word occures we mask and evalute each word occurance seperately
    true_sentence_list  = true_sentence.split()
    indices = [i for i, x in enumerate(true_sentence_list) if x == word]
    for i in indices:
        true_sentence_list_copy = true_sentence_list.copy()
        # get number of tokens after tokenizing the word
        n_tokens_input = len(tokenizer.tokenize(word))
        # replace the exact word by the [MASK] tag for each token (subwords)
        true_sentence_list_copy[i] = " ".join(["[MASK]"]*n_tokens_input)
        masked_sentence = " ".join(true_sentence_list_copy) #re.sub(r"\b"+word+r"\b", " ".join(["[MASK]"]*n_tokens_input), true_sentence)
        # get the top predicted word and the loss value
        (predicted, loss_actual) = getLoss(true_sentence, masked_sentence)

        
        
        # # true sentence with the true predicted words
        # true_sentence2 = true_sentence
        # masked_sentence2 = true_sentence
        # true_sentence2 = re.sub(r"\b"+word+r"\b", pred,true_sentence2, count=1)
        n_tokens_predicted =  len(tokenizer.tokenize(predicted))
        # masked_sentence2 = re.sub(r"\b"+pred+r"\b", "[MASK]",true_sentence2, count=1)# true_sentence2.replace(pred, " ".join(["[MASK]"]*n_tokens_irrelevant))

        # get the loss for the predicted word
        true_sentence_list_copy[i] = " ".join(["[MASK]"]*n_tokens_predicted)
        masked_sentence2 = " ".join(true_sentence_list_copy)
        (predicted, loss_relevant) = getLoss(true_sentence, masked_sentence2)

        # get the top 100 predicted words for each mask
        pred_top_100 = getTopPred(masked_sentence)
        # get random word which doesn't exists in the top 100
        irrelevant_word = random.choices(list(tokenizer.vocab.keys()- (pred_top_100.split()+["[UNK]"," ",''])), k=1)[0]    
    

        # n_tokens_pred = len(tokenizer.tokenize(predicted))
        # sentence_predicted_word =  re.sub(r"\b"+word+r"\b", predicted, true_sentence)
        # (predicted, loss_relevant) = getLoss(sentence_predicted_word, masked_sentence = re.sub(r"\b"+predicted+r"\b", " ".join(["[MASK]"]*n_tokens_pred), sentence_predicted_word))
    
        # get the loss for the irrelevant word
        n_tokens_irrelevant = len(tokenizer.tokenize(irrelevant_word))
        true_sentence_list_copy[i] = irrelevant_word
        sentence_irrelevant_word = " ".join(true_sentence_list_copy) # re.sub(r"\b"+word+r"\b", irrelevant_word, true_sentence)

        true_sentence_list_copy[i] = " ".join(["[MASK]"]*n_tokens_irrelevant)
        masked_sentence3 =  " ".join(true_sentence_list_copy)
        (predicted, loss_not_relevant) = getLoss(sentence_irrelevant_word, masked_sentence3)

        
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
        report.append((pred, actual_relevant_precent, actual_notrelevant_precent, predicted, irrelevant_word))
    return report
