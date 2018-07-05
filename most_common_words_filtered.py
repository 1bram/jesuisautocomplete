#JE SUIS AUTOCOMPLETE CORPUS ANALYSE

import itertools
import csv
from nltk import FreqDist
from nltk.corpus import stopwords

######### CORPUS EINLESEN
autocompleted = []

with open("jesuisautocomplete_ger.csv") as csvfile:
       reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) 
       for row in reader: # JEDE ZEILE IST EIN AUTO VERVOLLSTÄNDIGTER SATZ
           autocompleted.append(row)

l_autocompleted = list(itertools.chain.from_iterable(autocompleted))

str_autocompleted = " ".join(l_autocompleted)

tkn_autocompleted = str_autocompleted.split() # TOKENS BILDEN
##########

german_stopwords = set(stopwords.words("german")) # NLTK LIBRARY STOPWORDS
#print (german_stopwords)

english_stopwords = set(stopwords.words("english")) # NLTK LIBRARY STOPWORDS

arr_custom_stopwords = []

with open("custom_stopwords_new.csv") as csvfile:
       reader2 = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) 
       for row2 in reader2: # JEDE ZEILE IST EIN AUTO VERVOLLSTÄNDIGTER SATZ
           arr_custom_stopwords.append(row2)

custom_stopwords =set(itertools.chain.from_iterable(arr_custom_stopwords))

all_stopwords = german_stopwords | english_stopwords | custom_stopwords

#print(all_stopwords)

fltrd_autocompleted = []

for w in tkn_autocompleted:
    if w not in all_stopwords:
        fltrd_autocompleted.append(w)

frequency = FreqDist(fltrd_autocompleted)
frequency2 = FreqDist(tkn_autocompleted)

#print (frequency2.most_common(100))
print (frequency.most_common(100))





