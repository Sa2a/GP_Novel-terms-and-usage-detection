
from transformers import BertTokenizer, BertForMaskedLM
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained("UBC-NLP/MARBERT",local_files_only=False)
model = BertForMaskedLM.from_pretrained("UBC-NLP/MARBERT",local_files_only=False)

def getLoss(true_sentence ,masked_sentence, return_all_pred = False):
    #  tokenize the input into ides and add the suitable tags
    inputs = tokenizer(masked_sentence, return_tensors="pt")
    # get only the tokenized input ides
    labels = tokenizer(true_sentence, return_tensors="pt")["input_ids"]

    # mask labels of non-[MASK] tokens -100 for the cross intropy loss to ignore the evaluation
    labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

    outputs = model(**inputs, labels=labels)
    # logits returns the output of the mode with shape (1, number of tokens, 100000 prediction for each token), ex. torch.Size([1, 9, 100000])
    with torch.no_grad():
        logits = outputs.logits
    # retrieve array of indexes of [MASK] in the sentence, ex. tensor([1, 5])
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    # get the maximum prediction value for each mask, ex. tensor([1932, 2079])
    predicted_token_id = logits[0, mask_token_index]

    # seperate each predicted mask word in an element in the array accending order 
    # predicted_words = [tokenizer.decode(id) for id in predicted_token_id]
    return (tokenizer.decode(predicted_token_id.argmax(axis=1)), round(outputs.loss.item(), 2), predicted_token_id if return_all_pred else None)

# def getTopPred(masked_sentence,topN = 100):
#     inputs = tokenizer(masked_sentence, return_tensors="pt")
#     with torch.no_grad():
#         logits = model(**inputs).logits
#     mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
#     predicted_token_id = logits[0, mask_token_index]    
#     topN_tokend_id = np.argpartition(predicted_token_id.reshape(-1), -topN)[-topN:]
#     return tokenizer.decode(topN_tokend_id)

def getTopPred(predicted_token_id, topN = 100):
    # return the topN for each word in predicted_token_id
    each_top = []
    for token_ides in predicted_token_id.detach().numpy():
        each_top.append(tokenizer.decode(np.argpartition(token_ides, -topN)[-topN:]))
    return each_top

def getAllPred_ides(masked_sentence):
    inputs = tokenizer(masked_sentence, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index]    
    return predicted_token_id

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
def anyIn(list1 , list2):
    for x in list1:
        if x in list2:
            return True
    return False

def getRelevanceScore(true_sentence_list, word, word_index):
    # report = []

    # indices of the word in case of many of the same word occures we mask and evalute each word occurance seperately
    true_sentence  = " ".join(true_sentence_list)
    # true_sentence_list  = true_sentence.split()
    word_tokens = tokenizer.tokenize(word)
    # indices = [i for i, x in enumerate(true_sentence_list) if x == word]
    # for i in indices:

    i = word_index
    true_sentence_list_copy = true_sentence_list.copy()
    # get number of tokens after tokenizing the word
    n_tokens_input = len(word_tokens)
    # replace the exact word by the [MASK] tag for each token (subwords)
    true_sentence_list_copy[i] = " ".join(["[MASK]"]*n_tokens_input)
    masked_sentence = " ".join(true_sentence_list_copy) #re.sub(r"\b"+word+r"\b", " ".join(["[MASK]"]*n_tokens_input), true_sentence)
    # get the top predicted word and the loss value
    (predicted, loss_actual, predicted_token_id) = getLoss(true_sentence, masked_sentence, return_all_pred= True)

    
    
    # # true sentence with the true predicted words
    # true_sentence2 = true_sentence
    # masked_sentence2 = true_sentence
    # true_sentence2 = re.sub(r"\b"+word+r"\b", pred,true_sentence2, count=1)
    n_tokens_predicted =  len(tokenizer.tokenize(predicted))
    # masked_sentence2 = re.sub(r"\b"+pred+r"\b", "[MASK]",true_sentence2, count=1)# true_sentence2.replace(pred, " ".join(["[MASK]"]*n_tokens_irrelevant))

    # get the loss for the predicted word
    true_sentence_list_copy[i] = " ".join(["[MASK]"]*n_tokens_predicted)
    masked_sentence2 = " ".join(true_sentence_list_copy)

    true_sentence_list_copy[i] = predicted
    sentence_pred = " ".join(true_sentence_list_copy)

    (predicted2, loss_relevant, non) = getLoss(sentence_pred, masked_sentence2)

    # get the top 100 predicted words for each mask
    # pred_top_100 = getTopPred(predicted_token_id)

    topN = 100000
    pred_top_100000 = getTopPred(predicted_token_id, topN)
    # get random word which doesn't exists in the top 100000 
    irrelevant_word = " ".join(pred_top_100000).split()[-1]
    # excluded_list = list(tokenizer.vocab.keys()- (" ".join(pred_top_100000).split()+["[UNK]"," ",'',"##"]))
    # while irrelevant_word in excluded_list or  len(tokenizer.tokenize(irrelevant_word)) >1:
    #     irrelevant_word = random.choices(excluded_list, k=1)[0]    


    # n_tokens_pred = len(tokenizer.tokenize(predicted))
    # sentence_predicted_word =  re.sub(r"\b"+word+r"\b", predicted, true_sentence)
    # (predicted, loss_relevant) = getLoss(sentence_predicted_word, masked_sentence = re.sub(r"\b"+predicted+r"\b", " ".join(["[MASK]"]*n_tokens_pred), sentence_predicted_word))

    # get the loss for the irrelevant word
    n_tokens_irrelevant = len(tokenizer.tokenize(irrelevant_word))
    true_sentence_list_copy[i] = irrelevant_word
    sentence_irrelevant_word = " ".join(true_sentence_list_copy) # re.sub(r"\b"+word+r"\b", irrelevant_word, true_sentence)

    true_sentence_list_copy[i] =" ".join(["[MASK]"]*n_tokens_irrelevant)
    masked_sentence3 =  " ".join(true_sentence_list_copy)
    (predicted3, loss_not_relevant,non) = getLoss(sentence_irrelevant_word, masked_sentence3)

    
    point = 0
    # if word in " ".join(pred_top_100):
    #     point = 0.5* loss_actual
    # elif word in " ".join(pred_top_100000) or anyIn(word_tokens," ".join(pred_top_100000)):
    #     point = 0.2* loss_actual
    # # elif anyIn(word_tokens," ".join(pred_top_100000)):
    # #     point = 0.2* loss_actual
    # else:
    #     point = -0.5* loss_actual
    
    # print(point)
    loss_actual = loss_actual- point
    loss_not_relevant = loss_not_relevant + point
    
    actual_relevant_dist = abs( loss_actual - loss_relevant  )
    actual_not_relevant_dist = abs( loss_actual - loss_not_relevant )

    total_distance  = actual_relevant_dist + actual_not_relevant_dist

    actual_relevant_precent = actual_not_relevant_dist  / total_distance
    actual_notrelevant_precent = actual_relevant_dist / total_distance

    # pred = 0 if actual_relevant_precent >= .2 else 1
    # report.append((pred, actual_relevant_precent, predicted, irrelevant_word))
    return (actual_relevant_precent, predicted, irrelevant_word)
