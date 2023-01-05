import numpy as np
from flask import Flask, request, render_template
import pickle 


# !pip install ar-corrector
# !pip install pyarabic

import pickle
import re
import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import numpy as np
from nltk.corpus import stopwords
import os
import warnings
import itertools
from nltk.corpus import stopwords
import string
# from ar_corrector.corrector import Corrector


def removeshortsen(txt):
  words=txt.split(' ')
  text=''
  if len(words) > 4:
    text = ' '.join(words)

  return text

def removeMention(txt):
  arabic_men=re.sub("@[أ-ي]+","",txt)#//remove Arabic hashtags that have  after it
  arabic_men=re.sub("[أ-ي]@+","",arabic_men) #remove Arabic hashtags that have # before it
  without_men=re.sub("@[A-Za-z]+","",arabic_men)# remove English hashtags
  return without_men

def removeduplicate(txt):
  non_duplicate = ''.join(i for i, _ in itertools.groupby(txt))
  return non_duplicate

def reduce_characters(inputText):
    '''
    step #4: Reduce character repitation of > 2 characters at time
              For example: the word 'cooooool' will convert to 'cool'
    '''
    # pattern to look for three or more repetitions of any character, including
    # newlines.
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    reduced_text = pattern.sub(r"\1\1", inputText)
    return reduced_text

def removeDigits(txt):
  without_digit = re.sub('[0-9]+', '', txt) 
  return without_digit

def removeURl(txt):
  without_url= re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', txt)
  return without_url

def removestopwords(txt):
  without_stopwords = ' '.join(word for word in txt.split() if word not in stopwords)
  return without_stopwords

def removepuncatution(txt):

  punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation
  translator = str.maketrans('', '', punctuations)
  without_punc = txt.translate(translator)

  # txtt = re.sub("[\s+\\# ـ!\/_,$%=^*?:@&^~`(+\"]+|[+！，。？、~@￥%……&*（）“ ”:;：；、\\《）《》“”()»〔〕# ]+ ·.『]", "", txt)
  # seq = re.sub("[\s+[`÷×؛<>_()*&% ـ،/:{} ÷×؛<>_()*&^%][ـ،/: ]", "", txtt)
  return without_punc

def removenonrabic(txt):
  # Remove None arabic 
  without_english = re.sub(u"[^\u0621-\u063A\u0640-\u0652 ]", " ", txt)
  return without_english

def removeHashtages(txt):
  arabic_hash=re.sub("#[أ-ي]+","",txt)#//remove Arabic hashtags that have  after it
  arabic_hash_=re.sub("[أ-ي]#+","",arabic_hash) #remove Arabic hashtags that have # before it
  without_hash=re.sub("#[A-Za-z]+","",arabic_hash)# remove English hashtags
  return without_hash

def removetashkeel(txt):
  TATWEEL = u"\u0640"
  p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
  without_tashkeel = re.sub(p_tashkeel,"", txt)
  without_tashkeel=without_tashkeel.replace(TATWEEL, '')
  return without_tashkeel

def remove_Emojis(text):
  emoj = re.compile("["
      u"\U0001F600-\U0001F64F"  # emoticons
      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
      u"\U0001F680-\U0001F6FF"  # transport & map symbols
      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
      u"\U00002500-\U00002BEF"  # chinese char
      u"\U00002702-\U000027B0"
      u"\U00002702-\U000027B0"
      u"\U000024C2-\U0001F251"
      u"\U0001f926-\U0001f937"
      u"\U00010000-\U0010ffff"
      u"\u2640-\u2642" 
      u"\u2600-\u2B55"
      u"\u200d"
      u"\u23cf"
      u"\u23e9"
      u"\u231a"
      u"\ufe0f"  # dingbats
      u"\u3030"
                    "]+", re.UNICODE)
  without_emojy=re.sub(emoj, '', text)
  return without_emojy


def Clean_Egyptain_text(raw_text):

  raw_text=raw_text.strip()

  #1 remove shorten sentences that contain less than 4 words 
  # text=removeshortsen(raw_text)

  #2 remove duplichated chars in the raw text
  #text=removeduplicate(text)
  text=reduce_characters(raw_text)

  #3 remove hashtages after remove duplicated chars
  text=removeHashtages(text)

  #4 remove digits
  text= removeDigits(text)

  #5 remove the puncatution from the text 
  text=removepuncatution(text)

  #6 remove the mention and @ char
  text=removeMention(text)

  #7 remove arabic tashkeel from the text
  text=removetashkeel(text)

  #8 remove Emojies from the text
  text=remove_Emojis(text)

  #9 remove url and links from the text
  text=removeURl(text)

  # #9 remove stopwords from the text
  # text=removestopwords(text)

  #10 remove non rabic words from the text
  text=removenonrabic(text)


 #10 remove non rabic words from the text
  # text=correction(text)

  return text



# !pip install transformers


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

################################################################################################################################################################


import random
import re 
def anyIn(list1 , list2):
    for x in list1:
        if x in list2:
            return True
    return False

def getRelevanceScore(true_sentence_list, word, word_index):


    # indices of the word in case of many of the same word occures we mask and evalute each word occurance seperately
    true_sentence  = " ".join(true_sentence_list)
    # true_sentence_list  = true_sentence.split()
    word_tokens = tokenizer.tokenize(word)
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

    n_tokens_predicted =  len(tokenizer.tokenize(predicted))
    # masked_sentence2 = re.sub(r"\b"+pred+r"\b", "[MASK]",true_sentence2, count=1)# true_sentence2.replace(pred, " ".join(["[MASK]"]*n_tokens_irrelevant))

    # get the loss for the predicted word
    true_sentence_list_copy[i] = " ".join(["[MASK]"]*n_tokens_predicted)
    masked_sentence2 = " ".join(true_sentence_list_copy)

    true_sentence_list_copy[i] = predicted
    sentence_pred = " ".join(true_sentence_list_copy)

    (predicted2, loss_relevant, non) = getLoss(sentence_pred, masked_sentence2)



    topN = 100000
    pred_top_100000 = getTopPred(predicted_token_id, topN)
    # get random word which doesn't exists in the top 100000 
    irrelevant_word = " ".join(pred_top_100000).split()[-1]

    # get the loss for the irrelevant word
    n_tokens_irrelevant = len(tokenizer.tokenize(irrelevant_word))
    true_sentence_list_copy[i] = irrelevant_word
    sentence_irrelevant_word = " ".join(true_sentence_list_copy) # re.sub(r"\b"+word+r"\b", irrelevant_word, true_sentence)

    true_sentence_list_copy[i] =" ".join(["[MASK]"]*n_tokens_irrelevant)
    masked_sentence3 =  " ".join(true_sentence_list_copy)
    (predicted3, loss_not_relevant,non) = getLoss(sentence_irrelevant_word, masked_sentence3)

    point = 0
    loss_actual = loss_actual- point
    loss_not_relevant = loss_not_relevant + point
    
    actual_relevant_dist = abs( loss_actual - loss_relevant  )
    actual_not_relevant_dist = abs( loss_actual - loss_not_relevant )

    total_distance  = actual_relevant_dist + actual_not_relevant_dist

    actual_relevant_precent = actual_not_relevant_dist  / total_distance
    actual_notrelevant_precent = actual_relevant_dist / total_distance

    return (actual_relevant_precent, predicted, irrelevant_word)
#####################################################################################################################################################

import pyarabic.araby as araby
import numpy as np
from numpy.linalg import norm
class ArabicCorrector:
    n_char = 29
    myCharDictionary = araby.ALPHABETIC_ORDER.copy()
    myCharDictionary.update({'ى': myCharDictionary['ي']})
    myCharDictionary.update({'آ': myCharDictionary['ا']})
    myCharDictionary.update({'أ': myCharDictionary['ا']})
    myCharDictionary.update({'إ': myCharDictionary['ا']})


    def getPositionEncoding(seq_len=200, d=300, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P
    
    p_encoding = getPositionEncoding()
    p_average = np.average(p_encoding,axis=1)
    empty_vector  = [1]* n_char
    def __init__(self, correct_words,init = 1) :
        self.correct_words = correct_words
        

    def myWord2vec_arabic(word):
        vector  = [0]*ArabicCorrector.n_char
        for c in word:
            try:
                vector[ArabicCorrector.myCharDictionary[c]-1]+=1
            except:
                pass
        return vector
    def cosin_similarity_arabic( word1, word2):
        # print(word1,word2)
        A = np.array(ArabicCorrector.myWord2vec_arabic(word1))
        B = np.array(ArabicCorrector.myWord2vec_arabic(word2))

        # compute cosine similarity
        cosine = np.dot(A,B)/(norm(A, axis=0)*norm(B))
        return cosine

   
    
    def myWord2vec_ave(word):
        vector = ArabicCorrector.empty_vector.copy()
        for index_word, c in enumerate(word):
            try:
                vector[ArabicCorrector.myCharDictionary[c]-1]*=ArabicCorrector.p_average[index_word]
            except:
                pass
        return vector

    def cosin_similarity_postional_embedding_ave( word1, word2):
        A = ArabicCorrector.myWord2vec_ave(word1)
        B = ArabicCorrector.myWord2vec_ave(word2)
        cosine = np.dot(A,B)/(norm(A, axis=0)*norm(B))
        return cosine

    def get_most_similar(self,word,metric = cosin_similarity_arabic, tolerance = 1):
        matched_word  =[]
        max_s = 0
        myTolearnce = tolerance
        for w in self.correct_words:
            cos = metric(word,w)
            if cos >= max_s:
                if myTolearnce ==0:
                    matched_word.append((cos,w))
                    matched_word.sort(key= lambda record:record[0])
                    matched_word.pop(0)
                    max_s = matched_word[0][0]
                    continue
                matched_word.append((cos,w))
                myTolearnce -=1
        matched_word.sort(key= lambda record:record[0],reverse=True)
        return matched_word
    def get_most_similar_distance(self,word,metric, tolerance = 1):

        matched_word  =[]
        max_s = np.inf
        myTolearnce = tolerance
        for w in self.correct_words:
            dis = metric(word,w)
            if dis <= max_s:
                if myTolearnce ==0:
                    matched_word.append((dis,w))
                    matched_word.sort(key= lambda record:record[0],reverse=True)
                    matched_word.pop(0)
                    max_s = matched_word[0][0]
                    continue
                matched_word.append((dis,w))
                myTolearnce -=1
        matched_word.sort(key= lambda record:record[0],reverse=False)
        return matched_word

import numpy as np

class Syntactic_Symilarity:
    def __init__(self, data):
        self.corpus = data
    def jaccard_set(self, a, b):
        # convert to set
        a = set(a)
        b = set(b)
        # calucate jaccard similarity
        j = float(len(a.intersection(b))) / len(a.union(b))
        return j

    def syntactic_similarity(self, symp_t):
        most_sim=[]
        for symp in self.corpus:
            d=self.jaccard_set(symp_t,symp)
            most_sim.append(d)
        
        order=np.argsort(most_sim)[::-1].tolist()
        matched_word=[(most_sim[order[0]], self.corpus[order[0]])]
        for i in range(1,len(order)):
            if most_sim[order[i-1]] != most_sim[order[i]]:
                break
            matched_word.append((most_sim[order[i]], self.corpus[order[i]]))
        return matched_word


def correct(word, MARBert_trie, topK = 1 ):
    half = word[:len(word)//2]
    pre_suf = MARBert_trie.prefixes(half)+ MARBert_trie.keys(half)

    if word in MARBert_trie:
        return  True
    else:
        # word = re.sub(r"ة$","ه",word)
        # word = re.sub(r"آ$","ء",word)
        # word = re.sub(r"أ$","ء",word)
        # word = re.sub(r"إ$","ء",word)
        # if word in MARBert_trie:
        #     corrected_5.append(word)
        #     continue
        syntactic2  = Syntactic_Symilarity(pre_suf)
        corrected_neighbor = syntactic2.syntactic_similarity(word)
        if corrected_neighbor[0][0] == 1:
            words = [wo[1] for wo in corrected_neighbor]
            temp  = ArabicCorrector(words)
            return temp.get_most_similar(word,metric=ArabicCorrector.cosin_similarity_postional_embedding_ave,tolerance= topK)
        syntactic = Syntactic_Symilarity(MARBert_trie.keys())
        corrected_all = syntactic.syntactic_similarity(word)
        words = [wo[1] for wo in corrected_all]
        temp  = ArabicCorrector(words)
        return temp.get_most_similar(word,metric=ArabicCorrector.cosin_similarity_postional_embedding_ave,tolerance= topK)

######################################################################################################################################################
# !pip install marisa_trie
import sys
import pyarabic.araby as araby
import marisa_trie


MARBert_trie = marisa_trie.Trie().load("MARBert_trie_dictionary.marisa")


def scan_text(text):
    sentences = araby.sentence_tokenize(text)
    x = araby.sentence_tokenize(sentences[-1])
    while len(x)>1:
        x = araby.sentence_tokenize(sentences.pop())
        sentences.extend(x)
    report =[]
    for sentence in sentences:
        sentence = Clean_Egyptain_text(sentence)
        sentenceList = sentence.split()
        for i, word in enumerate(sentenceList):
            corrected = correct(word,MARBert_trie)
            relevance_no_correction = getRelevanceScore(sentenceList,word, i)
            if corrected == True:
                relevance_with_correction = np.nan
            else:
                corrected = corrected[0][1]
                sentenceList[i] = corrected
                relevance_with_correction = getRelevanceScore(sentenceList,corrected, i)[0]
                sentenceList[i] = word
                # if relevance_with_correction <= relevance_no_correction[0] :
                #     sentenceList[i] = word
                # sentence = " ".join(sentenceList)
            report.append((sentence,i,word,relevance_no_correction[0],corrected,relevance_with_correction,relevance_no_correction[1],relevance_no_correction[2]))
    return pd.DataFrame(report,columns =["sentence","index","word","relevance_no_correction","corrected","relevance_with_correction","predicted_word","irrelevant_word"])

pickle.dump(model, open("model.pkl",'wb'))

app= Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def homepage():
    return render_template('index.html',**locals())
@app.route('/predict', methods = ['POST','GET'])
def predict():
    text=request.form['texta']    
    data= scan_text(text)
    #for sentence in result2:
     #   result+=sentence
    return render_template('index.html',**locals(),   tables=[data.to_html(classes='data')], titles=data.columns.values)
        
if __name__ == "__main__":
    app.run(debug=True,port=8000)
