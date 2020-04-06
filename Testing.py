import pandas
import re
import itertools
from collections import Counter
from collections import defaultdict
import numpy as np

def tokenize(sent):
    # input: a single sentence as a string.
    # output: a list of each "word" in the text
    # must use regular expressions

    # Retain capitalization.
    # Separate punctuation from words, except for (a) abbreviations of capital letters (e.g. “U.S.A.”), (b) hyphenated words (e.g. “data-driven”) and contractions (e.g. “can’t”).
    # Allow for hashtags and @mentions as single words (e.g. “#sarcastic”, “@sbunlp”)

    # <FILL IN>

    pattern = r"\d|[\#\@]?(?:[a-zA-Z]\-)+\w|</s>|<s>|<newline>|[\.\,\!\?\;]|[\.\.\.]|[\#\@]?(?:[A-Z]\.)+|[\#\@]?\w+[\-]\w+|[#\@]?[^\[^\(^:^\s^)^\]][A-z]*|[#\@\$]?\d+[\.\%\,]?\d*|[\-]"
    # [\#\@]?(?:[A-Z]\.?){2,} ---> Abbreviations
    # [\#\@]?\w+[\'\-]\w+ --->  Contractions and hyphens
    # [#\@]?[A-z]+  ---> Any word
    # [#\@]?\$?\d+[\.\%]?\d* ---> Percentages, Dollar Currency, Numbers, Decimal
    # [#\@]?[^\s]+ ---> Any combination of non-space characters 1 or more times
    # [\.\:\,\!\?\;] ----> Punctuation
    # [\.\.\.] -----> Ellipses
    # <\s> && <\s> -----> Starting and Ending of Sentences 
    # <newline> ----> \n replacement
    # \[\w+.\] ----> [Verse:]
    # \d ----> Single Digits

    tokens = re.findall(pattern, sent)

    return tokens

df = pandas.read_csv('songdata.csv') # Reading the csv file 
df = df.rename(columns={'text': 'lyrics' })        
df['artist-song'] = df[['artist', 'song']].agg('-'.join, axis=1)       
df['artist-song'] = df['artist-song'].map(lambda x: x.replace(" ", "_"))
df['artist-song'] = df['artist-song'].map(lambda x: x.lower())
df['lyrics'] = df['lyrics'].map(lambda x: '<s> ' + x + ' </s>')
df['lyrics'] = df['lyrics'].map(lambda x: x.replace("\n", " <newline> "))
df = df.set_index('artist-song')   


#################################################################
# 2. Tokenization: Tokenize The Song Title By Appling Lambda function To All Elements In The Song Column
#################################################################

def generateLyrics(tokensList, wordsInVocab):          
  
    bigramCounts = makeBigram(tokensList)
    trigramCounts = makeTrigram(tokensList)
    
    lyrics = ["<s>"]
    
    for i in range(30): 
        wordProbability = ({'a': 1})       
        if i == 0:
            wordProbability = wordProbs( ("<s>",) , bigramCounts, trigramCounts, wordsInVocab )
        else:                 
            wordProbability = wordProbs( (lyrics[i-1], lyrics[i]) , bigramCounts, trigramCounts, wordsInVocab )
       
        words = list(wordProbability.keys())
        probs = list(wordProbability.values())      
         
        norm = [float(i)/sum(probs) for i in probs]
        word = ""
        while True:
            wordinList = np.random.choice(words, 1, p=norm)        
            word = str(wordinList[0]) 
            if (word != "<OOV>"):
                break   
        lyrics.append(word)
    
    lyrics.append("</s>")
    
    return lyrics

def buildVocab(words):
    lyricsToList = list(itertools.chain.from_iterable(words)) # Chaining All The List of Lists 
    wordToIndex = dict((Counter(lyricsToList)))  # Make The Dictionary 
    wordsInVocab = dict(filter(lambda x: x[1] > 2, wordToIndex.copy().items()))
    wordsOOV = dict(filter(lambda x: x[1] <= 2, wordToIndex.copy().items()))   
    wordsInVocab.update({'<OOV>': len(wordsOOV)})
    
    return wordsInVocab , wordsOOV

def find_ngrams(input_list, n):
  return list(zip(*[input_list[i:] for i in range(n)]))

def makeDoubleDictBigram(bigramCount, wordToIndex):
    
    bigramCounts = defaultdict(dict)
    
    for i in bigramCount:       
       previousWord = i[0]
       currentWord = i[1]       
       if (previousWord not in wordToIndex):
           bigramCounts["<OOV>"][currentWord] = bigramCount[i]
       elif (currentWord not in wordToIndex):
           bigramCounts[previousWord]["<OOV>"] = bigramCount[i]
       else:
           bigramCounts[previousWord][currentWord] = bigramCount[i]
    
    return bigramCounts

def makeBigram(tokenList):
    bigramList = list(map(lambda x: find_ngrams(x,2), tokenList )) #Generting a list of bigrams for all the song lyrics
    bigramListCount = Counter(itertools.chain.from_iterable(bigramList)) # Count All The Bigram Occurences
    bigramCount = dict(bigramListCount.items()) # Make a dict with all the bigrams and their counts     
    bigramCounts = makeDoubleDictBigram(bigramCount, wordsInVocab) # Make a double dict based on the bigram and their counts
    return bigramCounts

def makeDoubleDictTrigram(trigramCount, wordToIndex):
    
    trigramCounts = defaultdict(dict)
    
    for i in trigramCount:             
       firstWord, secondWord, thirdWord = i
       if (firstWord not in wordToIndex):
           firstWord = "<OOV>"
       if (secondWord not in wordToIndex):
           secondWord = "<OOV>" 
       if (thirdWord not in wordToIndex):
           thirdWord = "<OOV>"
       previousBigram = (firstWord, secondWord)
       trigramCounts[previousBigram][thirdWord]  = trigramCount[i] 
    
    return trigramCounts

def makeTrigram(tokenList):
    trigramList = list(map(lambda x: find_ngrams(x,3), tokenList ))
    trigramListCount = Counter(itertools.chain.from_iterable(trigramList))   
    trigramCount = dict(trigramListCount.items())  
    trigramCounts = makeDoubleDictTrigram(trigramCount, wordsInVocab)  # Make a double dict based on the trigram and their counts
    return trigramCounts

 
def getAddKSmoothBigram(currentWord, previousWord, bigram, wordToIndex):
    
    if (previousWord not in wordToIndex):
        previousWord = "<OOV>"
    probablity = (bigram[previousWord][currentWord] + 1) / (wordToIndex[previousWord] + len(wordToIndex))
    
    return probablity

def getAddKSmoothTrigram(currentWord, wordIminus1, wordIminus2, bigram, trigram, wordToIndex ):
    
    if (wordIminus1 not in wordToIndex):
        wordIminus1 = "<OOV>"
    if (wordIminus2 not in wordToIndex):
        wordIminus2 = "<OOV>"
                
    # Check if trigramCount is not None
    trigramCount = trigram.get( (wordIminus2, wordIminus1))
    
    if trigramCount is None:
        trigramCount = 0
    else:
        trigramCount = trigram.get( (wordIminus2, wordIminus1)).get(currentWord)
        if trigramCount is None:
            trigramCount = 0   
            
    bigramCount1 = bigram.get( wordIminus1 ).get(wordIminus2)
    if (bigramCount1 is None):
        bigramCount1 = 0
    bigramCount2 = bigram[wordIminus1][currentWord]
    if (bigramCount2 is None):
        bigramCount1 = 0
    
    countWordIminus1 = wordToIndex[wordIminus1]    
    vocabSize = len(wordToIndex)    
    trigramProbability = (trigramCount + 1) / (bigramCount1 + vocabSize )
    bigramProbability = (bigramCount2 + 1) / (countWordIminus1 + vocabSize)
    interpolatedProbability = (trigramProbability + bigramProbability) / 2
    
    return interpolatedProbability
 
def wordProbs(previousWords, bigram, trigram, wordToIndex): 
    
    probabilties_list_dict = dict()
    if len(previousWords) == 2:
        wordIminus1 = previousWords[1]
        wordIminus2 = previousWords[0]
        potential_words_i = list(bigram[wordIminus1].keys())
        potential_words_filter = list(filter(lambda x: x in wordsInVocab, potential_words_i)) #Filter Out OOV words for w_i
        if (potential_words_filter is []):
            probablity = wordToIndex[wordIminus1] / sum(wordToIndex.values())   
            probabilties_list_dict.update({wordIminus1: probablity})
        else:
            probabilties_list = list(map(lambda x: getAddKSmoothTrigram(x, wordIminus1, wordIminus2, bigram, trigram, wordToIndex) , potential_words_filter))
            probabilties_list_dict = dict(zip(potential_words_filter, probabilties_list))
    elif len(previousWords) == 1:
        wordIminus1 = previousWords[0]
        potential_words_i = list(bigram[wordIminus1].keys())
        potential_words_filter = list(filter(lambda x: x in wordToIndex, potential_words_i)) #Filter Out OOV words for w_i
        if (potential_words_filter is []):
            probablity = wordToIndex[wordIminus1] / sum(wordToIndex.values())  
            probabilties_list_dict.update({wordIminus1: probablity})         
        else: # There are wi with bigrams with word wi-1, then
            probabilties_list = list(map(lambda x: getAddKSmoothBigram(x, wordIminus1, bigram, wordToIndex) , potential_words_filter))            
            probabilties_list_dict = dict(zip(potential_words_filter, probabilties_list)) 
    elif isinstance(previousWords, str):
        wordIminus1 = previousWords
        potential_word_i = list(bigram[wordIminus1].keys())
        potential_words_filter = list(filter(lambda x: x in wordToIndex, potential_words_i)) #Filter Out OOV words for w_i
        if (potential_words_filter is []):
            probablity = wordToIndex[wordIminus1] / sum(wordToIndex.values())   
            probabilties_list_dict.update({wordIminus1: probablity})         
        else: # There are wi with bigrams with word wi-1, then
            probabilties_list = list(map(lambda x: getAddKSmoothBigram(x, wordIminus1, bigram, wordToIndex) , potential_words_filter))            
            probabilties_list_dict = dict(zip(potential_words_filter, probabilties_list)) 
        
    return probabilties_list_dict
    
     
df['title_tokenized'] = df['song'].map(lambda x: tokenize(x.lower()))
df['lyrics_tokenized'] = df['lyrics'].map(lambda x : tokenize(x.lower()))

tokensList = df['lyrics_tokenized'][0:500]
wordsInVocab, wordsOOV = buildVocab(df['lyrics_tokenized'][0:500])

lyrics = generateLyrics(tokensList, wordsInVocab)
print(lyrics)