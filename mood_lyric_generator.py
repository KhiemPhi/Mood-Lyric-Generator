# CSE354 Sp20; Assignment 2 Version v01
##################################################################
_version_ = 0.1

import sys
import os
import re
import pandas
import numpy as np
from collections import Counter
from collections import defaultdict
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

##################################################################
# Tokenizer For Song Titles

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

#################################################################

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
      
def getConllTags(filename):
    #input: filename for a conll style parts of speech tagged file
    #output: a list of list of tuples
    #        representing [[[word1, tag1], [word2, tag2]]]
    wordTagsPerSent = [[]]
    sentNum = 0
    with open(filename, encoding='utf8') as f:
        for wordtag in f:
            wordtag=wordtag.strip()
            if wordtag:#still reading current sentence
                (word, tag) = wordtag.split("\t")
                wordTagsPerSent[sentNum].append((word,tag))
            else:#new sentence
                wordTagsPerSent.append([])
                sentNum+=1
    return wordTagsPerSent

def zero_signals (length):
    return [0] * length

def signal_flipper (index, zero_signals):
    zero_signals[index] = 1
    return zero_signals

def vowel_counter (word):
    # Counting Consonants and Vowels
    c_count = 0
    v_count = 0 
    vowels = ('a', 'e', 'i', 'o', 'u', 'y', 'A', 'E', 'I', 'O', 'U', 'Y')
    for i in word:
        if i in vowels:
                v_count = v_count + 1
        elif ((i >= 'a' and i <= 'z') or (i >= 'A' and i <= 'Z')):
                c_count = c_count + 1

    
    return c_count, v_count

def adjust_signal(index, index_count, vector, length, tokens, wordToIndex, num_words, adjectivesList, nounsList,verbsList):
    
    wordRegular = tokens[index + index_count]
    wordNext = ""
    wordBefore = ""
    if (index + index_count + 1 < num_words):
        wordNext = tokens[index + index_count + 1]
    if (index + index_count - 1 >= 0):
        wordBefore = tokens[index + index_count - 1]
   
    word = tokens[index + index_count].lower()  
    dictIndex = wordToIndex.get(word)
    
    c_count, v_count = vowel_counter(word)
    
    if dictIndex is not None:
        signal_flipper(dictIndex, vector)
    else:            
        signal_flipper(length-1, vector)
    
    #Add c_count, v_count to word: Feature 1
    pronouns = ["he", "she", "it", "they", "I", "you"]
    verbToBe = ["is", "am", "are", "'s", "'re", "'m"]
    suffix = ["al", "ial", "ial", "ical", "able", "ible", "an", "ian", "ary", "full", "ic", "ive", "ish", "less", "like", "y", "ous", "ose", "ant", "ent", "ile"]
    prefix = ["a", "ab", "anti", "counter", "de", "dis", "hyper", "il", "im", "in", "inter", "ir", "mis", "non", "over", "pre", "sub", "super", "un", "under"]
    colors = ['red', 'green', 'blue', 'yellow','pink', 'purple', 'black', "white", "gray", "violet"]
    adjectivesList = adjectivesList + colors
    
    vector[length- 2] = c_count 
    vector[length- 3] = v_count
    vector[length - 4] = 10000 if wordRegular in adjectivesList else 0
    vector[length - 5] = 100 if wordNext.lower() in nounsList or wordNext.lower() in pronouns else 0
    vector[length - 6] = 10 if wordBefore.lower() in verbToBe  else 0
    
    for i in suffix:
        suffix_len = len(i)
        checker = wordRegular.lower()
        if checker[-suffix_len:] == i:
            vector[length - 7] = 1000
            break
        
    for i in prefix:
        prefix_len = len(i)
        checker = wordRegular.lower()
        if checker[:prefix_len] == i:
            vector[length - 8] = 1000
            break
    
    
    vector[length - 9] = 1 if wordRegular.lower() not in wordToIndex or wordRegular not in wordToIndex else 0
    vector[length - 10] = 1 if wordBefore.lower() not in wordToIndex or wordBefore not in wordToIndex else 0
    vector[length - 11] = 1 if wordNext.lower() not in wordToIndex or wordNext not in wordToIndex else 0
    
    return vector    
    

def getFeaturesForTokens(tokens, wordToIndex, adjectivesList, nounsList,verbsList):
    #input: tokens: a list of tokens,
    #wordToIndex: dict mapping 'word' to an index in the feature list.
    #output: list of lists (or np.array) of k feature values for the given target

    num_words = len(tokens)
    featuresPerTarget = list() #holds arrays of feature per word
    
    for targetI in range(num_words):                     
        # Setting up 3 Encoders  
        length = len(wordToIndex) + 12 # Adding 1 to be OOD Detector , Add 2 for Vowels and Consonant Count (length-1, length-3)
        word_after_encoder = zero_signals(length)
        word_before_encoder = zero_signals(length)
        word_current_encoder = zero_signals(length)
        word_iminus2_encoder = zero_signals(length)
        word_iplus2_encoder = zero_signals(length)
        word_iminus3_encoder = zero_signals(length)
        word_iplus3_encoder = zero_signals(length)        
              
        word_current_encoder = adjust_signal(targetI, 0, word_current_encoder, length, tokens, wordToIndex, num_words, adjectivesList, nounsList,verbsList)        
       
        if (targetI + 1 < num_words):         
            word_after_encoder = adjust_signal(targetI, 1, word_after_encoder, length, tokens, wordToIndex, num_words, adjectivesList, nounsList,verbsList)
                    
        if (targetI > 0):    
            word_before_encoder = adjust_signal(targetI, -1, word_before_encoder, length, tokens, wordToIndex, num_words, adjectivesList, nounsList,verbsList) 
        
        if (targetI + 2 < num_words):         
            word_iplus2_encoder = adjust_signal(targetI, 2, word_iplus2_encoder, length, tokens, wordToIndex, num_words, adjectivesList, nounsList,verbsList) 
        
        if (targetI - 2 >= 0):    
            word_iminus2_encoder = adjust_signal(targetI, -2, word_iminus2_encoder, length, tokens, wordToIndex, num_words, adjectivesList, nounsList,verbsList) 
                
        #if (targetI + 3 < num_words):         
            #word_iplus3_encoder = adjust_signal(targetI, 3, word_iplus3_encoder, length, tokens, wordToIndex, num_words, adjectivesList, nounsList,verbsList) 
        
        #if (targetI - 3 >= 0):    
           # word_iminus3_encoder = adjust_signal(targetI, -3, word_iminus3_encoder, length, tokens, wordToIndex, num_words, adjectivesList, nounsList,verbsList)               
        
           
        featuresPerTarget.append(word_iminus2_encoder + word_before_encoder + word_current_encoder + word_after_encoder + word_iplus2_encoder) 

    return featuresPerTarget #a (num_words x k) matrix

def trainAdjectiveClassifier(features, adjs):
    #inputs: features: feature vectors (i.e. X)
    #        adjs: whether adjective or not: [0, 1] (i.e. y)
    #output: model -- a trained sklearn.linear_model.LogisticRegression object
    
    #<FILL IN>

    #Splitting our feature (X_train) and adjs (y_train) into development set 
    X_trainsub, X_dev, y_trainsub, y_dev = train_test_split(features, adjs, 
                                                  test_size=0.10, random_state=42)
    
    #Defining Penalties And Accuracy And Model:
    
    model = LogisticRegression(C=10, penalty="l1", solver='liblinear')
    model.fit(X_trainsub, y_trainsub)
    
    return model

def buildVocab(words):
    lyricsToList = list(itertools.chain.from_iterable(words)) # Chaining All The List of Lists 
    wordToIndex = dict((Counter(lyricsToList)))  # Make The Dictionary 
    wordsInVocab = dict(filter(lambda x: x[1] > 2, wordToIndex.copy().items()))
    wordsOOV = dict(filter(lambda x: x[1] <= 2, wordToIndex.copy().items()))   
    wordsInVocab.update({'<OOV>': len(wordsOOV)})
    
    return wordsInVocab , wordsOOV

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

def getMoodLyrics (adjective, artist_song_dict):
    titles = artist_song_dict.get(adjective)
    listofTokens = []
    for i in titles:
        listofTokens.append(df["lyrics_tokenized"][i])
    return listofTokens

def lyricsGenerator (adjective, artist_song_dict, wordsInVocab):
    lyrics = []
    if (artist_song_dict.get(adjective) is None):
        print(adjective + "is not detected as an adjective. Can't generate song lyrics")
        return lyrics
    else:
       tokens = getMoodLyrics(adjective, artist_song_dict)
       lyrics = generateLyrics(tokens, wordsInVocab)
    return lyrics
    
# Main
if __name__== '__main__':       
    #################################################################
    # 1. Preprocessing: (a) Replace The Column Index With Artist-Song After Concatenating The Strings And Set Them To Lower Case
    #                   (b) Adding Special Tokens (<s>) To Beginning And (</s>) At The End Of Song Lyrics 
    #                   (c) Replacing All NewLines with "<newline>"
    #################################################################
     
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
    
     
    df['title_tokenized'] = df['song'].map(lambda x: tokenize(x.lower()))
    df['lyrics_tokenized'] = df['lyrics'].map(lambda x : tokenize(x.lower()))
    
     
    #################################################################
    # Checkpoint 1: Print the tokenized title and lyrics for the follow artist-songs: 
    #  (1)  abba-burning_my_bridges
    #  (2)  beach_Boys-do_you_remember?
    #  (3)  avril_Lavigne-5,_4,_3,_2,_1
    #  (4)  michael_Buble-l-o-v-e
    #################################################################
    
    print("Checkpoint 1: ")
    print ("Tokenized Titles:")
    print(df['title_tokenized']['abba-burning_my_bridges'])
    print(df['title_tokenized']['beach_boys-do_you_remember?'])
    print(df['title_tokenized']["avril_lavigne-5,_4,_3,_2,_1_(countdown)"])
    print(df['title_tokenized']["michael_buble-l-o-v-e"])   

    print ("Tokenized Lyrics:")
    print(df['lyrics_tokenized']['abba-burning_my_bridges'])      
    print(df['lyrics_tokenized']['beach_boys-do_you_remember?'])
    print(df['lyrics_tokenized']["avril_lavigne-5,_4,_3,_2,_1_(countdown)"])
    print(df['lyrics_tokenized']["michael_buble-l-o-v-e"])
    

    #################################################################
    # 3. Creating A Vocabulary Of Words From Lyrics    
    #################################################################
    
    wordsInVocab, wordsOOV = buildVocab(df['lyrics_tokenized'][0:5000])
    
    #################################################################
    # 4. Creating A Bigram And Trigram Matrix
    #################################################################    
    
    bigramCounts = makeBigram(df["lyrics_tokenized"][0:5000])
    trigramCounts = makeTrigram(df["lyrics_tokenized"][0:5000])  
    
    #################################################################
    # Checkpoint 2: Based on just the first 5,000 lyrics, print the following (add-1 smoothed) probabilities:
    #  (1)  p(wi = “you”| wi-2 = “I”, wi-1 = “love”.) 
    #  (2)  p(wi=”special| wi-1=”midnight”)
    #  (3)  p(wi=”special”| wi-1="very")      
    #  (4)  p(wi=”special”| wi-2=”something ”,wi-1=”very”)
    #  (5)  p(wi=”funny”| wi-2=”something ”,wi-1=”very”)    
    #################################################################
    
    print("Checkpoint 2: ")
    trigramProbsILove = wordProbs( ("i", "love") , bigramCounts, trigramCounts, wordsInVocab )
    
    print(  "p(wi = 'you' | wi-2 = 'I', wi-1 = 'love') = %.5f " % trigramProbsILove["you"] )
    
    bigramProbsSpecial = wordProbs( ("midnight",) , bigramCounts, trigramCounts, wordsInVocab )
    print(  "p(wi = 'special' | wi-1 = 'midnight') = %.5f "   %  bigramProbsSpecial["special"]  )
    
    bigramProbsVery = wordProbs( ("very",) , bigramCounts, trigramCounts, wordsInVocab )
    print("p(wi = 'special'| wi-1 = 'very') = %.5f " % bigramProbsVery["special"] )  
    
    trigramProbsSomethingVery = wordProbs( ("something", "very") , bigramCounts, trigramCounts, wordsInVocab )
    print("p(wi = 'special'| wi-2 = 'something' , wi-1 = 'very') = %.5f " % trigramProbsSomethingVery["special"] )
    print("p(wi = 'funny'| wi-2 = 'something' , wi-1 = 'very') = %.5f " % trigramProbsSomethingVery["funny"])
    
    #################################################################
    # 5. Training Model Based On Previous Code
    #################################################################
    taggedSents = getConllTags('daily547.conll')
    
    
    #Getting Adjectives 
    taggedSents_flat = list(itertools.chain.from_iterable(taggedSents))  
    adjectives = list(filter(lambda x: x[1] == 'A' ,taggedSents_flat.copy()))
    adjectivesList , tags1 = zip(*adjectives)
    adjectivesList = list(adjectivesList)
    
    #Getting Nouns
    
    nouns = list(filter(lambda x: x[1] == 'N' ,taggedSents_flat.copy()))
    nounsList , tags2 = zip(*nouns)
    nounsList = list(nounsList)
    
    # Getting Verbs
    verbs = list(filter(lambda x: x[1] == 'V' ,taggedSents_flat.copy()))
    verbsList , tags2 = zip(*nouns)
    verbsList = list(verbsList)    
    
    
    print("\n[ Feature Extraction Test ]\n")
    #first make word to index mapping: 
    wordToIndexTagged = set() #maps words to an index
    for sent in taggedSents:
        if sent:
            words, tags = zip(*sent) #splits [(w, t), (w, t)] into [w, w], [t, t]
            wordToIndexTagged |= set([w.lower() for w in words]) #union of the words into the set
    print("  [Read ", len(taggedSents), " Sentences]")
    #turn set into dictionary: word: index
    wordToIndexTagged = {w: i for i, w in enumerate(wordToIndexTagged)}
    
    #Next, call Feature extraction per sentence
    sentXs = []
    sentYs = []
    print("  [Extracting Features]")
    for sent in taggedSents:        
        if sent:
            words, tags = zip(*sent)            
            sentXs.append(getFeaturesForTokens(words, wordToIndexTagged, adjectivesList, nounsList,verbsList)) 
            sentYs.append([1 if t == 'A' else 0 for t in tags])
    
    print("\n[ Classifier Test ]\n")       
    #flatten by word rather than sent: 
    X = [j for i in sentXs for j in i]    
    y= [j for i in sentYs for j in i]
    try: 
        X_train, X_test, y_train, y_test = train_test_split(np.array(X),
                                                            np.array(y),
                                                            test_size=0.20,
                                                            random_state=42)
    except ValueError:
        print("\nLooks like you haven't implemented feature extraction yet.")
        print("[Ending test early]")
        sys.exit(1)
    print("  [Broke into training/test. X_train is ", X_train.shape, "]")
    #Train the model.
    print("  [Training the model]")
    tagger = trainAdjectiveClassifier(X_train, y_train) # Model To Tag Adjectives
    print("  [Done]")
    
    song_titles = list(df["title_tokenized"])
    adjectives = []
    artist_song_dict = dict()
    
    print("[Identify Adjectives In Song Titles:]")
    
    for i in range(len(song_titles)):
        counter = i+1
        title = song_titles[i]        
        vectorTitle = getFeaturesForTokens(title, wordToIndexTagged, adjectivesList, nounsList,verbsList)
        y_pred = tagger.predict(vectorTitle)
        adjective_list = dict(zip(title, y_pred))
        adjectives_from_list = list(filter(lambda x: adjective_list[x] == 1, adjective_list.copy() ))
        # With these adjectives, we add the dict mapping
        for adj in adjectives_from_list:
            if counter < len(song_titles):
                value = df["title_tokenized"][i:counter].index.values
            if artist_song_dict.get(adj) is None:                
                artist_song_dict.update({adj: []})
                artist_song_dict[adj].append(value[0])
            else:
                artist_song_dict[adj].append(value[0])            
        
        adjectives = adjectives + adjectives_from_list
    print("     [Done]")    
        
    final_adjective_list = set(adjectives) #---> All The Adjectives
    
    final_adjective_counter = dict(Counter(adjectives))    
       
    artist_song_dict = dict(filter(lambda x: len(x[1]) >= 10 , artist_song_dict.items() )) #---> Filter out adjectives that occurs < 10 times
    
    print("Checkpoint 3:")
    print()
    print("Artist Song: Good")
    print(artist_song_dict.get("good"))
    print("Artist Song: Happy")
    print(artist_song_dict.get("happy"))
    print("Artist Song: Afraid")
    print(artist_song_dict.get("afraid"))
    print("Artist Song: Red")
    print(artist_song_dict.get("red"))
    print("Artist Song: Blue")
    print(artist_song_dict.get("blue"))
      
    
    
    print(artist_song_dict.get('red'))
    print ("Lyrics Based On: Red")
    print(lyricsGenerator("red", artist_song_dict, wordsInVocab))
    
   
    
   
    
  
    
    
   
    
    
    

       

   

   

