import re
import os
import sys
import copy
import random
import pickle
import gensim
import warnings
import collections
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score,f1_score,accuracy_score

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

def read_pid_aid(infile):
    AllRecords_original = []
    with open(infile, 'r', encoding = 'utf8') as f:
        for line in f:
            read_data = line.split("\t")
            # get ride of bad formated lines
            if(len(read_data)==13 or len(read_data)==12):
                paper_detail = {"paperID": read_data[0], "authorID":read_data[1]}
                AllRecords_original.append(paper_detail)
            else:
                print(len(read_data))
        f.close()
    return pd.DataFrame(AllRecords_original)

def read_text_embedding(Dataset = "pubmed", emb_type = "off", training_size = "3m", labeled_only = True):
    if training_size == "140k":
        labeled_only = False
    text_emb = []
    emb_pid = []
    while True:
        if emb_type == "tf":
            modelSaveDir = "../models/"+Dataset+"/tf/text_sample="+training_size+"/"
            with open(modelSaveDir+'tf_features.pickle', "rb") as input_file:
                vec = pickle.load(input_file)
            with open(modelSaveDir+'feature_pid.pickle', "rb") as input_file:
                allPaperid = pickle.load(input_file)
            text_emb = vec.toarray()
            emb_pid = allPaperid
            break
        elif emb_type == "tf_idf":
            modelSaveDir = "../models/"+Dataset+"/tf_idf/text_sample="+training_size+"/"
            with open(modelSaveDir+'tf_idf_trained_features.pickle', "rb") as input_file:
                vec = pickle.load(input_file)
            with open(modelSaveDir+'feature_pid.pickle', "rb") as input_file:
                allPaperid = pickle.load(input_file)
            text_emb = vec.toarray()
            emb_pid = allPaperid
            break
        elif emb_type == "lsa":
            if labeled_only:
                modelSaveDir = "../vectors/"+Dataset+"/"+emb_type+"/text_sample=3m/"
                with open(modelSaveDir+"extracted_labeled_lsa.txt", 'r', encoding = 'utf8') as f:
                    for line in f:
                        read_data = line.split(" ")
                        paper_Vectors = read_data[1:]
                        emb_pid.append(read_data[0])
                        text_emb.append(paper_Vectors)
                break
            else:
                modelSaveDir = "../models/"+Dataset+"/lsa/text_sample="+training_size+"/"
                with open(modelSaveDir+'lsa_Matrix.pickle', "rb") as input_file:
                    vec = pickle.load(input_file)
                with open(modelSaveDir+'feature_pid.pickle', "rb") as input_file:
                    allPaperid = pickle.load(input_file)
                text_emb = vec
                emb_pid = allPaperid
                break
        elif emb_type == "pv_dm":
            if labeled_only:
                modelSaveDir = "../vectors/"+Dataset+"/"+emb_type+"/text_sample=3m/"
                with open(modelSaveDir+"extracted_labeled_pv_dm.txt", 'r', encoding = 'utf8') as f:
                    for line in f:
                        read_data = line.split(" ")
                        paper_Vectors = read_data[1:]
                        emb_pid.append(read_data[0])
                        text_emb.append(paper_Vectors)
                break
            else:
                modelSaveDir = "../models/"+Dataset+"/"+emb_type+"/text_sample="+training_size+"/"
                model = gensim.models.Doc2Vec.load(modelSaveDir+"Doc2Vec(dmm,d100,n5,w5,mc2,s0.001,t24)")
                allPaperTags = model.docvecs.offset2doctag
                for pid in allPaperTags:
                    vectorRepresentation = model.docvecs[pid].tolist()
                    vectorRepresentation = [float(i) for i in vectorRepresentation]
                    text_emb.append(vectorRepresentation)
                    emb_pid.append(pid)
                break
        elif emb_type == "pv_dbow":
            if labeled_only:
                modelSaveDir = "../vectors/"+Dataset+"/"+emb_type+"/text_sample=3m/"
                with open(modelSaveDir+"extracted_labeled_pv_dbow.txt", 'r', encoding = 'utf8') as f:
                    for line in f:
                        read_data = line.split(" ")
                        paper_Vectors = read_data[1:]
                        emb_pid.append(read_data[0])
                        text_emb.append(paper_Vectors)
                break
            else:
                modelSaveDir = "../models/"+Dataset+"/"+emb_type+"/text_sample="+training_size+"/"
                model = gensim.models.Doc2Vec.load(modelSaveDir+"Doc2Vec(dbow,d100,n5,mc2,s0.001,t24)")
                allPaperTags = model.docvecs.offset2doctag
                for pid in allPaperTags:
                    vectorRepresentation = model.docvecs[pid].tolist()
                    vectorRepresentation = [float(i) for i in vectorRepresentation]
                    text_emb.append(vectorRepresentation)
                    emb_pid.append(pid)
                break
        elif emb_type == "off":
            break
        else:
            print("Embedding type not available, selecting default setting")
            emb_type="off"
    if emb_type != "off":
        print("Total text vector records:",len(text_emb))
        print("Vector dimension: ", len(text_emb[0]))
    return text_emb, emb_pid

# read trained paper citation graph
def read_citation_embedding_sorted(Dataset = "pubmed", emb_type = "off", labeled_only = True):
    
    citation_emb = []
    while True:
        if emb_type == "n2v":
            if labeled_only:
                citation_emb_dir = "../vectors/"+Dataset+"/"+emb_type+"/extracted_labeled_n2v.txt"
                with open(citation_emb_dir, 'r', encoding = 'utf8') as f:
                    for line in f:
                        read_data = line.split(" ")
                        if(len(read_data)==101):
                            citation_emb.append(read_data)
                f.close()
                break
            else:
                citation_emb_dir = "../vectors/"+Dataset+"/"+emb_type+"/n2v.txt"
                with open(citation_emb_dir, 'r', encoding = 'utf8') as f:
                    for line in f:
                        read_data = line.split(" ")
                        if(len(read_data)==101):
                            citation_emb.append(read_data)
                f.close()
                break
        elif emb_type =="node2vec":
            if labeled_only:
                citation_emb_dir = "../vectors/"+Dataset+"/"+emb_type+"/extracted_labeled_node2vec.txt"
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
    if emb_type != "off":
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

def extract_embedding(all_embedding, all_embedding_pid, wanted_pid_list):
    extracted_emb = []
    wanted_pid_list = wanted_pid_list.values.tolist()
    total_missing_sample = 0
    # only if embedding exist
    if len(all_embedding)>0:
        # loop through wanted pid list to keep input order
        for wanted_pid in wanted_pid_list:
            # if wanted paper in all pretrained embeddings
            if wanted_pid in all_embedding_pid:
                emb_idx = all_embedding_pid.index(wanted_pid)
                extracted_emb.append(all_embedding[emb_idx])
            # if wanted paper not in all pretrained embeddings, fill missing sample with 0's
            else:
                total_missing_sample+=1
                print("Missing Sample: ", wanted_pid)
                temp = [0] * len(all_embedding[0])
                extracted_emb.append(temp)
    print("Total missing sample: ", total_missing_sample)
    extracted_emb = pd.DataFrame(extracted_emb)
    return extracted_emb

def extract_sorted_embedding(all_embedding, wanted_pid_list):
    extracted_emb = []
    wanted_pid_list = wanted_pid_list.values.tolist()
    wanted_pid_list = [int(x) for x in wanted_pid_list]
    wanted_pid_list = list(sorted(set(wanted_pid_list)))
    total_missing_sample = 0
    # only if embedding exist
    if len(all_embedding)>0:
        # loop through wanted pid list to keep input order
        for embedding in all_embedding:
            if(len(wanted_pid_list)==0):
                break
            while (wanted_pid_list[0]<=int(embedding[0])):
                if wanted_pid_list[0]==int(embedding[0]):
                    extracted_emb.append(embedding)
                    wanted_pid_list.remove(int(embedding[0]))
                elif (wanted_pid_list[0]<int(embedding[0])):
                    total_missing_sample+=1
                    # ------------------------ fill it up with 0's -------------------------- #
                    fill_na = [wanted_pid_list[0]]
                    temp = [0] * (len(all_embedding[0])-1)
                    final_filled_zero_emb = fill_na+temp
                    extracted_emb.append(final_filled_zero_emb)
                    # ----- or do nothing and remove those missing samples from dataset ----- #
                    # remove paper that not in all dataset
                    wanted_pid_list.remove(wanted_pid_list[0])
                if len(wanted_pid_list)==0:
                    break
    print("Total missing sample: ", total_missing_sample)
    extracted_emb = pd.DataFrame(extracted_emb)
    return extracted_emb

# collect unlabeled vectors
def extract_unlabeled_embedding(allembedding, unlabeled_pid):
    unlabeled_pid = [int(x) for x in unlabeled_pid]
    unlabeled_pid = list(sorted(set(unlabeled_pid)))
    wanted_embedding = []
    for embedding in allembedding:
        if(len(unlabeled_pid)==0):
            break
        while (unlabeled_pid[0]<=int(embedding[0])):
            if unlabeled_pid[0]==int(embedding[0]):
                wanted_embedding.append(embedding)
                unlabeled_pid.remove(int(embedding[0]))
            elif (unlabeled_pid[0]<int(embedding[0])):
                # remove paper that not in all dataset
                unlabeled_pid.remove(unlabeled_pid[0])
            if len(unlabeled_pid)==0:
                break
    unlabeled_data = pd.DataFrame(wanted_embedding)
    unlabeled_data['label'] = "-1"
    unlabeled_data = unlabeled_data.rename(columns={0: 'paperID'})
    return unlabeled_data

# remove author(positive sample) from other(negative sample)
def extractNegativeSample(positiveSample, allSample):
    negativeSample = [x for x in allSample if x not in positiveSample]
    return negativeSample

# synchronize data wrt pid
def synchro_views(labeled_dv1, labeled_dv2, unlabeled_data1, unlabeled_data2):
    noCitationPids_labeled = set(labeled_dv1[0])-set(labeled_dv2[0])
    print("labeled no citation link: ", len(noCitationPids_labeled))
    noCitationPids_unlabeled = set(unlabeled_data1["paperID"])-set(unlabeled_data2["paperID"])
    print("Unlabeled no citation link size: ", len(noCitationPids_unlabeled))
    # process unlabeled data
    unlabeled_dv1 = unlabeled_data1[~unlabeled_data1["paperID"].isin(noCitationPids_unlabeled)].reset_index(drop=True)
    unlabeled_dv2 = unlabeled_data2
    # process labeled data
    labeled_dv1_final = labeled_dv1[~labeled_dv1[0].isin(noCitationPids_labeled)].reset_index(drop=True)
    labeled_dv2_final = labeled_dv2.reset_index(drop=True)
    # since our input data are sorted, all data are in order with pid
    return labeled_dv1_final, labeled_dv2_final, unlabeled_dv1, unlabeled_dv2

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

def normal_predict_with_accumulate_statistic(X_train, y_train, X_test, y_test, clf, verbose=True):
    clf.fit(X_train, y_train)
    # get predicted label
    label_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,label_pred)
    f1 = f1_score(y_test, label_pred,average='macro')
    
    # accumulate statistic for entire model f1
    cnf_matrix = metrics.confusion_matrix(y_test, label_pred)
    TP = np.diag(cnf_matrix)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    
    if verbose:
        print(metrics.classification_report(y_test, label_pred))
        print(metrics.confusion_matrix(y_test, label_pred).ravel())
        print(cnf_matrix)
        print("TP: ",TP, "TN: ",TN, "FP: ",FP,"FN: ",FN)
    return accuracy, f1, TP.sum(), TN.sum(), FP.sum(), FN.sum()

# cross validation
def k_fold_cv(data, label, clf, k=10, verbose=True):
    kf = StratifiedKFold(n_splits=k, shuffle=False)
    allTrueLabel = []
    allPredLabel = []
    for train_index, test_index in kf.split(data, label):
        # print("TRAIN:", train_index, " \n TEST:", test_index)
        # split train and test
        data_train, data_test = data[train_index], data[test_index]
        label_train, label_test = label[train_index], label[test_index]
        # fit data to clf
        per_fold_clf = copy.deepcopy(clf)
        per_fold_clf.fit(data_train, label_train)
        # get predicted label
        label_pred = per_fold_clf.predict(data_test)
        allTrueLabel.extend(label_test)
        allPredLabel.extend(label_pred)

    accuracy = accuracy_score(allTrueLabel, allPredLabel)
    f1 = f1_score(allTrueLabel, allPredLabel,average='macro')
    if verbose:
        print(metrics.classification_report(allTrueLabel, allPredLabel))
        print(metrics.confusion_matrix(allTrueLabel, allPredLabel).ravel())
    
    return accuracy, f1

def k_fold_cv_with_accumulate_statistic(data, label, clf, k=10, verbose=True):
    kf = StratifiedKFold(n_splits=k, shuffle=False)
    allTrueLabel = []
    allPredLabel = []
    for train_index, test_index in kf.split(data, label):
        # print("TRAIN:", train_index, " \n TEST:", test_index)
        # split train and test
        data_train, data_test = data[train_index], data[test_index]
        label_train, label_test = label[train_index], label[test_index]
        # fit data to clf
        per_fold_clf = copy.deepcopy(clf)
        per_fold_clf.fit(data_train, label_train)
        # get predicted label
        label_pred = per_fold_clf.predict(data_test)
        allTrueLabel.extend(label_test)
        allPredLabel.extend(label_pred)

    accuracy = accuracy_score(allTrueLabel, allPredLabel)
    f1 = f1_score(allTrueLabel, allPredLabel,average='macro')
    
    # accumulate statistic for entire model
    if len(set(label)) == 2:
        print("Binary case")
        TN, FP, FN, TP = confusion_matrix(allTrueLabel, allPredLabel).ravel()
        if verbose:
            print(metrics.classification_report(allTrueLabel, allPredLabel))
            print("TP: ",TP, "TN: ",TN, "FP: ",FP,"FN: ",FN)
    else:
        print("Multi-Class case")
        cnf_matrix = confusion_matrix(allTrueLabel, allPredLabel)
        TP = np.diag(cnf_matrix)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        if verbose:
            print(metrics.classification_report(allTrueLabel, allPredLabel))
            print(metrics.confusion_matrix(allTrueLabel, allPredLabel).ravel())
            print(cnf_matrix)
            print("TP: ",TP, "TN: ",TN, "FP: ",FP,"FN: ",FN)

    return accuracy, f1, TP.sum(), TN.sum(), FP.sum(), FN.sum()

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
            