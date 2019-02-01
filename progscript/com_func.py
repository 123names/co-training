import re
import os
import sys
import pickle
import gensim
import warnings
import collections
import numpy as np
import pandas as pd

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from multiprocessing.pool import ThreadPool, TimeoutError


# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()


def clean_line_of_raw(raw_text, lower = True, only_letter = True, word_min_length=2, word_max_length=50, stopword = True, stem_words = False):
    '''
    Must import some package
    import re
    from nltk.tokenize import RegexpTokenizer
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    '''
    # lower case the input
    if lower:
        raw_text = raw_text.lower()
    # define regular expression for different condition
    regexP1 = '[\w+-]'
    if only_letter:
        regexP1 = '[a-z]'
    # define tokenizer with re that remove character length lower than word_min_length
    tokenizer = RegexpTokenizer(regexP1+'{'+str(word_min_length)+',}')
    # Tokenize raw text
    doc_token = tokenizer.tokenize(raw_text)
    # remove token that have length more than word_max_length
    doc_token = [re.sub(r'[a-z]{'+str(word_max_length)+',}', '', word) for word in doc_token]
    # clean stop words
    if stopword:
        STOPWORDS = set(stopwords.words('english')) 
        cleaned_line = [t for t in doc_token if t not in STOPWORDS if t != '']
    else:
        cleaned_line = [t for t in doc_token if t != '']
    # stem words
    if stem_words:
        cleaned_line = [str(p_stemmer.stem(word)) for word in cleaned_line]
    
    return cleaned_line


def clean_batch_of_raw(raw_texts, stopword = True, only_letter = True, word_min_length=2,word_max_length=50, stem_words = False):
    cleaned_content = []
    row_sizes = []
    for line in raw_texts:
        cleaned_line = clean_line_of_raw(line, stopword=stopword,only_letter=only_letter, word_min_length=word_min_length,
                                                 word_max_length=word_max_length, stem_words=stem_words)
        cleaned_content.append(cleaned_line)
        row_sizes.append(len(cleaned_line))
    return cleaned_content, row_sizes

def read_textual_embedding(Dataset = "pubmed", emb_type = "off", training_size = "3m"):
    textual_emb = []
    emb_pid = []
    while True:
        if emb_type == "tf":
            modelSaveDir = "../../Data/"+Dataset+"/models/tf/textual_sample="+training_size+"/"
            with open(modelSaveDir+'tf_features.pickle', "rb") as input_file:
                vec = pickle.load(input_file)
            with open(modelSaveDir+'feature_pid.pickle', "rb") as input_file:
                allPaperid = pickle.load(input_file)
            textual_emb = vec.toarray()
            emb_pid = allPaperid
            break
        elif emb_type == "tf_idf":
            modelSaveDir = "../../Data/"+Dataset+"/models/tf_idf/textual_sample="+training_size+"/"
            with open(modelSaveDir+'tf_idf_trained_features.pickle', "rb") as input_file:
                vec = pickle.load(input_file)
            with open(modelSaveDir+'feature_pid.pickle', "rb") as input_file:
                allPaperid = pickle.load(input_file)
            textual_emb = vec.toarray()
            emb_pid = allPaperid
            break
        elif emb_type == "lsa":
            modelSaveDir = "../../Data/"+Dataset+"/models/lsa/textual_sample="+training_size+"/"
            with open(modelSaveDir+'lsa_Matrix.pickle', "rb") as input_file:
                vec = pickle.load(input_file)
            with open(modelSaveDir+'feature_pid.pickle', "rb") as input_file:
                allPaperid = pickle.load(input_file)
            textual_emb = vec
            emb_pid = allPaperid
            break
        elif emb_type == "pv_dm":
            if training_size == "3m":
                loadDir = "../../Data/"+Dataset+"/vectors/d2v/textual_sample=3m/extracted_labeled_pv_dm.txt"
                with open(loadDir, 'r', encoding = 'utf8') as f:
                    for line in f:
                        read_data = line.split(" ")
                        paper_Vectors = read_data[1:]
                        emb_pid.append(read_data[0])
                        textual_emb.append(paper_Vectors)
                f.close()
                break
            elif training_size == "140k":
                modelSaveDir = "../../Data/"+Dataset+"/models/doc2v/textual_sample="+training_size+"/"+emb_type+"/"
                model = gensim.models.Doc2Vec.load(modelSaveDir+"Doc2Vec(dmm,d100,n5,w5,mc2,s0.001,t24)")
                allPaperTags = model.docvecs.offset2doctag
                for pid in allPaperTags:
                    vectorRepresentation = model.docvecs[pid].tolist()
                    vectorRepresentation = [float(i) for i in vectorRepresentation]
                    textual_emb.append(vectorRepresentation)
                    emb_pid = allPaperTags
                break
                
        elif emb_type == "pv_dbow":
            if training_size == "3m":
                loadDir = "../../Data/"+Dataset+"/vectors/d2v/textual_sample=3m/extracted_labeled_pv_dm.txt"
                with open(loadDir, 'r', encoding = 'utf8') as f:
                    for line in f:
                        read_data = line.split(" ")
                        paper_Vectors = read_data[1:]
                        emb_pid.append(read_data[0])
                        textual_emb.append(paper_Vectors)
                f.close()
                break
            elif training_size == "140k":
                modelSaveDir = "../../Data/"+Dataset+"/models/doc2v/textual_sample="+training_size+"/"
                model = gensim.models.Doc2Vec.load(modelSaveDir+"pv_dbow/Doc2Vec(dbow,d100,n5,mc2,s0.001,t24)")
                allPaperTags = model.docvecs.offset2doctag
                for pid in allPaperTags:
                    vectorRepresentation = model.docvecs[pid].tolist()
                    vectorRepresentation = [float(i) for i in vectorRepresentation]
                    textual_emb.append(vectorRepresentation)
                    emb_pid = allPaperTags
                break
        elif emb_type == "off":
            break
        else:
            print("Embedding type not available, selecting default setting")
            emb_type="off"
    print("Total textual vector records:",len(textual_emb))
    print("Vector dimension: ", len(textual_emb[0]))
    return textual_emb, emb_pid

# read trained rec to rec node2vec citation graph
def read_citation_embedding(Dataset = "pubmed", emb_type = "off"):
    citation_emb = []
    emb_pid = []
    while True:
        if emb_type == "n2v":
            citation_emb_dir = "../../Data/"+Dataset+"/vectors/"+emb_type+"/extracted_labeled_n2v.txt"
            with open(citation_emb_dir, 'r', encoding = 'utf8') as f:
                for line in f:
                    read_data = line.split(" ")
                    if(len(read_data)==101):
                        emb_pid.append(read_data[0])
                        citation_emb.append(read_data[1:])
            f.close()
            break
        elif emb_type == "off":
            break
        else:
            print("Embedding type not available, selecting default setting")
            emb_type="off"
    print("Total citation vector records:",len(citation_emb))
    print("Vector dimension: ", len(citation_emb[0]))
    return citation_emb, emb_pid

# read trained rec to rec textual graph
def read_all_textual_embedding_sorted(Dataset = "pubmed", emb_type = "off", training_size = "3m"):
    textual_emb = []
    while True:
        if emb_type == "lsa":
            modelSaveDir = "../../Data/"+Dataset+"/models/lsa/textual_sample="+training_size+"/"
            with open(modelSaveDir+'lsa_Matrix.pickle', "rb") as input_file:
                vec = pickle.load(input_file)
            with open(modelSaveDir+'feature_pid.pickle', "rb") as input_file:
                allPaperid = pickle.load(input_file)
            allPaperid = np.array(allPaperid)
            textual_emb = np.column_stack((allPaperid,vec))
            break
        elif emb_type == "pv_dm":
            modelSaveDir = "../../Data/"+Dataset+"/models/doc2v/textual_sample="+training_size+"/"
            model = gensim.models.Doc2Vec.load(modelSaveDir+"pv_dm/Doc2Vec(dmm,d100,n5,w5,mc3,s0.001,t24)")
            textual_emb = []
            allPaperTags = model.docvecs.offset2doctag
            for pid in allPaperTags:
                vectorRepresentation = model.docvecs[pid].tolist()
                vectorRepresentation = [float(i) for i in vectorRepresentation]
                textual_emb.append([pid]+vectorRepresentation)
            break
        elif emb_type == "pv_dbow":
            modelSaveDir = "../../Data/"+Dataset+"/models/doc2v/textual_sample="+training_size+"/"
            model = gensim.models.Doc2Vec.load(modelSaveDir+"pv_dbow/Doc2Vec(dbow,d100,n5,mc3,s0.001,t24)")
            textual_emb = []
            allPaperTags = model.docvecs.offset2doctag
            for pid in allPaperTags:
                vectorRepresentation = model.docvecs[pid].tolist()
                vectorRepresentation = [float(i) for i in vectorRepresentation]
                textual_emb.append([pid]+vectorRepresentation)
            break
        elif emb_type == "off":
            break
        else:
            print("Embedding type not available, selecting default setting")
            emb_type="off"
    print("Total textual vector records:",len(textual_emb))
    print("Vector dimension: ", len(textual_emb[0]))
    textual_emb = sorted(textual_emb,key=lambda x: (int(x[0])))
    return textual_emb

# read trained rec to rec node2vec citation graph
def read_all_citation_embedding_sorted(Dataset = "pubmed", emb_type = "off"):
    citation_emb = []
    while True:
        if emb_type == "n2v":
            citation_emb_dir = "../../Data/"+Dataset+"/vectors/"+emb_type+"/n2v.txt"
            with open(citation_emb_dir, 'r', encoding = 'utf8') as f:
                for line in f:
                    read_data = line.split(" ")
                    if(len(read_data)==101):
                        citation_emb.append(read_data)
            f.close()
            break
        elif emb_type == "off":
            break
        else:
            print("Embedding type not available, selecting default setting")
            emb_type="off"
    print("Total citation vector records:",len(citation_emb))
    print("Vector dimension: ", len(citation_emb[0]))
    citation_emb = sorted(citation_emb,key=lambda x: (int(x[0])))
    return citation_emb

def select_productive_groups(labeled_data, threshold_select_name_group):
    # count number of paper each author write based on author ID
    authorCounter = collections.Counter(labeled_data["authorID"])
    # remove author that do not write more than 100 papers
    for k in list(authorCounter):
        if authorCounter[k] < threshold_select_name_group:
            del authorCounter[k]
    return authorCounter

def only_select_productive_authors(labeled_data, step_threshold):
    print("Total sample size before apply threshold: ",len(labeled_data))
    # count number of paper each author write based on author ID
    paperCounter = collections.Counter(labeled_data["authorID"])
    print(paperCounter)
    print("Total author before apply threshoid: ", len(paperCounter))
    # collect per class statistic
    for k in list(paperCounter):
        if paperCounter[k] < step_threshold:
            del paperCounter[k]
    temp =list(paperCounter.keys())
    print("Total author after apply threshoid: ", len(temp))
    # remove samples that are smaller than threshold
    labeled_data = labeled_data[labeled_data.authorID.isin(temp)]
    author_list = set(temp)
    print("Total sample size after apply threshold: ",len(labeled_data))
    return labeled_data, author_list, paperCounter

def productive_authors_list(labeled_data, step_threshold):
    # count number of paper each author write based on author ID
    paperCounter = collections.Counter(labeled_data["authorID"])
    print(paperCounter)
    print("Total author before apply threshoid: ", len(paperCounter))
    # collect per class statistic
    for k in list(paperCounter):
        if paperCounter[k] < step_threshold:
            del paperCounter[k]
    author_list = set(list(paperCounter.keys()))
    print("Total author after apply threshoid: ", len(paperCounter))
    return author_list
    
def merge_data_parts(part_collection):
    # remove empty emb (when emb set off)
    part_collection = [part for part in part_collection if len(part)!=0]
    print(len(part_collection))
    if len(part_collection)>1:
        combinedata = np.concatenate(part_collection,axis=1)
    elif len(part_collection)==1:
        if isinstance(part_collection[0], pd.DataFrame):
            combinedata = part_collection[0].values
        else:
            combinedata = part_collection[0]
    else:
        print("No data available")
        sys.exit()
    return combinedata


def process_input(prompt, answer):
    s = input(prompt)
    return s

def write_csv_df(savePath, filename, df):
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    # Give the filename you wish to save the file to
    pathfile = os.path.normpath(os.path.join(savePath,filename))

    # Use this function to search for any files which match your filename
    files_present = os.path.isfile(pathfile) 
    # if no matching files, write to csv, if there are matching files, print statement
    if not files_present:
        df.to_csv(pathfile, encoding='utf-8',index=False)
    else:
        threadp = ThreadPool(processes=1)
        overwrite = "y"
        prompt = "WARNING: " + pathfile + " already exists! Do you want to overwrite <y/n>? \n "
        try:
            overwrite = threadp.apply_async(process_input, args=(prompt, overwrite)).get(timeout=10)
        except TimeoutError: 
            print("No input found, overwrite old file")
        print(overwrite)
        if overwrite == 'y':
            print("reached")
            df.to_csv(pathfile, encoding='utf-8',index=False)
        elif overwrite == 'n':
            new_filename = input("Type new filename: \n ")
            write_csv_df(savePath,new_filename,df)
        else:
            print("Not a valid input. Data is NOT saved!\n")