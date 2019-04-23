Before running any code, download dataset (folder canopies) from ************ and other addition files (16May-IDs.txt, 16May-title.txt, 16May-abstract.txt, 16May-citation.txt) from **********

Notice data directory is should be looks like below:

progscript/
    A1_merge_rawfile.ipynb
    A1a_extract_textual (Generate file for training embedding).ipynb
    ... 
    Data/
        pubmed/
            allTextual/
                16May-IDs.txt
                16May-title.txt
                16May-abstract.txt
                16May-citation.txt
            canopies/
                canopy_a_aggarwal
                canopy_a_ali
                ... (Files omitted)
            ...

1. Generate our dataset (Textual part) from raw files (ALL data include unlabeled (total 3m papers))

Step 1: Run A1_merge_rawfile.ipynb to obtain merged raw file. 

Step 2: Run A1a_extract_textual (Generate file for training embedding).ipynb

Step 3: (optional) Run A1b_collect_data_statistic.ipynb

2. Generate our dataset (Textual part) from raw files (Labeled data only (total 140k papers))

Step 1: Run A2_extract_labeled.ipynb to obtain labeled only data (Create a new folder canopies_labeled)

Step 2: Run A2a_extract_labeled_textual (Generate file for training embedding).ipynb

Step 3: (optional) Run A2b_collect_labeled_statistic.ipynb

3. Use generated file (Textual part) to train embedding (Textual embedding):

Step 1: Run A3_train_embeddings.ipynb with (training_sample_size = "140k") for train labeled data only

Step 2: Run A3_train_embeddings.ipynb with (training_sample_size = "3m") for train all data

4. (optional) Extract embedding from all embedding for speed improvement (For embedding trained on 3m papers only)

Step 1: Run A4a_d2v_to_txt.ipynb to generate txt file contain all embeddings

Step 2: Run A4b_extract_labeled_emb.ipynb to generate txt file contain embedding trained on 3m paper, but only extract labeled ones (Only extract 140k from 3m)

5. The citation embedding we used directly from Yi. 

Step 1: Run B2_preprocess_sort_selected_emb.ipynb to extract embedding we needed (using pid)

6. Use embedding generated to train classifier (Labeled data only (total 140k papers))

Run files in folder 1_scikit_binary_selfdefine_ovr_global_140k_emb, title explained what script does.
Notice here co-training is using labeled data only, thus we assume some label are missing.

7. Use embedding generated to train classifier (ALL data include unlabeled (total 3m papers))

Same as 5, here co-training is using unlabeled data to augment labeled sample

