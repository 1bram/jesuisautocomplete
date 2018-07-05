import itertools
import nltk
from ClassifierBasedGermanTagger import ClassifierBasedGermanTagger
import pickle
import mysql.connector
import reprlib   

r = reprlib.Repr() # Beschränken der Ausgabe der String Länge 
r.maxlist = 200  
r.maxstring = 200
r.maxset = 100

db = mysql.connector.connect(host="81.169.151.194",user="instabilothek",password="sTealtHispAsswOrd",database="jesuisautocomplete", port="3306" )

cursor = db.cursor()

query = "SELECT DISTINCT results FROM jesuisautocomplete_ger WHERE LENGTH(results) > 30 ORDER BY RAND()"

cursor.execute(query)
rows = cursor.fetchall()

autocompleted = []

for row in rows: # each row is a list
    autocompleted.append(row)

l_autocompleted = list(itertools.chain.from_iterable(autocompleted))

with open('nltk_german_classifier_data.pickle', 'rb') as f:
    tagger = pickle.load(f)

def create_german_to_universal_dict():
    german_to_universal = {}
    german_to_universal[u"ADJA"]        = u"ADJ"
    german_to_universal[u"ADJD"]        = u"ADJ"
    german_to_universal[u"ADV"]         = u"ADV"
    german_to_universal[u"APPR"]        = u"ADV"
    german_to_universal[u"APPRART"]     = u"KON"    
    german_to_universal[u"APPO"]        = u"KON"
    german_to_universal[u"APZR"]        = u"KON"
    german_to_universal[u"ART"]         = u"KON"
    german_to_universal[u"CARD"]        = u"ZAHL"
    german_to_universal[u"FM"]          = u"NOMEN"
    german_to_universal[u"ITJ"]         = u"PT"
    german_to_universal[u"KON"]         = u"KON"
    german_to_universal[u"KOKOM"]       = u"KON"
    german_to_universal[u"KOUI"]        = u"KON"
    german_to_universal[u"KOUS"]        = u"KON"
    german_to_universal[u"NA"]          = u"NOMEN"
    german_to_universal[u"NE"]          = u"NOMEN"
    german_to_universal[u"NNE"]         = u"NOMEN"
    german_to_universal[u"NN"]          = u"NOMEN"
    german_to_universal[u"PAV"]         = u"PRON"
    german_to_universal[u"PAVREL"]      = u"KON"
    german_to_universal[u"PDAT"]        = u"KON"
    german_to_universal[u"PDS"]         = u"KON"
    german_to_universal[u"PIAT"]        = u"KON"
    german_to_universal[u"PIS"]         = u"PRON"
    german_to_universal[u"PPER"]        = u"PRON"
    german_to_universal[u"PRF"]         = u"KON"
    german_to_universal[u"PPOSS"]       = u"KON"
    german_to_universal[u"PPOSAT"]      = u"KON"
    german_to_universal[u"PRELAT"]      = u"KON"
    german_to_universal[u"PRELS"]       = u"KON"
    german_to_universal[u"PTKA"]        = u"KON"
    german_to_universal[u"PTKANT"]      = u"PT"
    german_to_universal[u"PTKNEG"]      = u"PTNEG"
    german_to_universal[u"PTKREL"]      = u"PT"
    german_to_universal[u"PTKVZ"]       = u"KON"
    german_to_universal[u"PTKZU"]       = u"KON"
    german_to_universal[u"PWS"]         = u"PRON"
    german_to_universal[u"PWAT"]        = u"KON"
    german_to_universal[u"PWAV"]        = u"KON"
    german_to_universal[u"PWAVREL"]     = u"KON"
    german_to_universal[u"PWREL"]       = u"KON"
    german_to_universal[u"VAFIN"]       = u"VERB"
    german_to_universal[u"VAIMP"]       = u"VERB"
    german_to_universal[u"VS"]          = u"VERB"
    german_to_universal[u"VAINF"]       = u"VERB"
    german_to_universal[u"VAPP"]        = u"VERB"
    german_to_universal[u"VMFIN"]       = u"VERB"
    german_to_universal[u"VS"]          = u"VERB"
    german_to_universal[u"VMINF"]       = u"VERB"
    german_to_universal[u"VPP"]         = u"VERB"
    german_to_universal[u"VMPP"]        = u"VERB"
    german_to_universal[u"VVFIN"]       = u"VERB"
    german_to_universal[u"VVIMP"]       = u"VERB"
    german_to_universal[u"VVINF"]       = u"VERB"
    german_to_universal[u"VVIZU"]       = u"VERB"
    german_to_universal[u"VVPP"]        = u"VERB"
    german_to_universal[u"VVPP"]        = u"VERB"
    german_to_universal[u"VPP"]         = u"VERB"
    german_to_universal[u"VPR"]         = u"VERB"
    german_to_universal[u"VS"]          = u"VERB"
    german_to_universal[u"XY"]          = u"NOMEN"
    german_to_universal[u"PROAV"]       = u"KON"
    german_to_universal[u"$("]          = u"."
    german_to_universal[u"$."]          = u"."
    german_to_universal[u"TRUNC"]       = u"???"
    german_to_universal[u"$,"]          = u"???"
    return german_to_universal

german_to_universal_dict = create_german_to_universal_dict()

def map_german_tag_to_universal(list_of_german_tag_tuples):
    return [ (tup[0], german_to_universal_dict[ tup[1] ]) for tup in list_of_german_tag_tuples ] 
    

def process_content():
    try:
        for i in l_autocompleted:
            words = nltk.word_tokenize(i)
            tagged = tagger.tag(words)
            universal_tagged = map_german_tag_to_universal(tagged)
            
            
    except Exception as e:
        print(str(e))
        
        
process_content()