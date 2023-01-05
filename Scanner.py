
import sys,os
from RelevanceScore import *
from Preprocessing_Detct_Novel_terms.preprocessing_detect_novel_terms2 import Clean_Egyptain_text
current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_directory+"\\Spelling  Correction")
from Misspelled_correction import *
import pyarabic.araby as araby
import marisa_trie
import pandas as pd
import numpy as np

MARBert_trie = marisa_trie.Trie().load(current_directory+"\\Data\\Dictionary\\MARBert_trie_dictionary.marisa")


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