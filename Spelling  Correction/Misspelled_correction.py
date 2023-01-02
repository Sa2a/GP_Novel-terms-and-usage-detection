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