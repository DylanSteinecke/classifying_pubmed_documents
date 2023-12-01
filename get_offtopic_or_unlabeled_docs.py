import numpy as np
import pandas as pd
import csv
import json
import os
import argparse

def get_offtopic_or_unlabeled_pmids(topic, num_random_pmids, max_pmid_number, get_offtopic_docs, get_unlabeled_docs):
    print('Getting the PMIDs of the "other" category documents...')
    feature_matrix_for_ontopic_categories_path = f'output/feature_matrix_{topic}.csv'
    topic_name = topic.split("_")[0]
    categories_of_interest_path = f'input/categories_list_of_list_of_tree_numbers_{topic_name}.json'
    pmids_of_categories_path = f'output/pmid_to_category_{topic}.json'
    pmids_not_of_categories_to_categories_path = f'output/pmid_to_category_less_than_{num_random_pmids}_non_{topic}.json'

    # Make PMID to label json
    if get_offtopic_docs:
        num_categories = len(json.load(open(categories_of_interest_path)))
        label = num_categories
    elif get_unlabeled_docs:
        label = -1
    random_pmids = np.random.randint(1, max_pmid_number, size=num_random_pmids).astype(str).tolist()
    pmids_of_categories = list(json.load(open(pmids_of_categories_path)).keys())
    pmids_not_of_categories = list(set(random_pmids).difference(set(pmids_of_categories)))
    pmids_not_of_categories_to_categories = {pmid:[label] for pmid in pmids_not_of_categories}
    with open(pmids_not_of_categories_to_categories_path,'w') as fout:
        json.dump(pmids_not_of_categories_to_categories, fout)
    print('Done!')
        
        
def get_offtopic_or_unlabeled_documents(num_random_pmids, topic, get_offtopic_docs, get_unlabeled_docs):
    '''Download random "other category" documents'''
    print('Downloading the "other" category documents...(output will be displayed at the end of function execution)')

    if get_offtopic_docs:
        pmid_to_offtopic_categories_path = f'output/pmid_to_category_less_than_{num_random_pmids}_non_{topic}.json'
        offtopic_feature_matrix_path = f'output/feature_matrix_less_than_{num_random_pmids}_non_{topic}.csv'
        os.system('python get_pubmed_docs.py --get_docs_on_pubmed '+\
                                          f'--pmid_to_cat {pmid_to_offtopic_categories_path} '+\
                                          f'--ft_mtrx_pth {offtopic_feature_matrix_path}'+\
                                          f'--get_offtopic_docs')
    elif get_unlabeled_docs:
        pmid_to_unlabeled_categories_path = f'output/pmid_to_category_less_than_{num_random_pmids}_unlabeled__({topic} study).json'
        unlabeled_feature_matrix_path = f'output/feature_matrix_less_than_{num_random_pmids}_unlabeled_({topic}).csv'
        os.system('python get_pubmed_docs.py --get_docs_on_pubmed '+\
                                          f'--pmid_to_cat {pmid_to_offtopic_categories_path} '+\
                                          f'--ft_mtrx_pth {offtopic_feature_matrix_path}'+\
                                          f'--get_unlabeled_topic_docs')
    print('Done!')
    
        
def merge_ontopic_and_offtopic_feature_matrices(topic, merge_matrix_option_1, merge_matrix_option_2):
    print('Combining the main categories and other category feature matrices...')
    feature_matrix_for_ontopic_categories_path = f'output/feature_matrix_{topic}.csv'
    num_topic_pmids = len(pd.read_csv(feature_matrix_for_ontopic_categories_path))
    feature_matrix_for_offtopic_category_path = f'output/feature_matrix_less_than_{num_random_pmids}_non_{topic}.csv'
    actual_num_random_pmids = len(pd.read_csv(feature_matrix_for_offtopic_category_path))
    combined_feature_matrix_path = f'output/feature_matrix_{actual_num_random_pmids}_non_{topic}_{num_topic_pmids}_{topic}.csv'

    if merge_matrix_option_1:
        '''
        Option 1: Load main and other feature matrices into memory and combine them.
        This alternates the rows between main and other categories, as long as there are
        enough rows from each matrix. Assumes there are more 'other' rows.
        '''
        ontopic_matrix = pd.read_csv(feature_matrix_for_ontopic_categories_path)
        other_matrix = pd.read_csv(feature_matrix_for_offtopic_category_path)

        with open(combined_feature_matrix_path, 'w') as fout:
            writer = csv.writer(fout)

            # Writes headers
            headers = ontopic_matrix.columns.tolist()
            writer.writerow(headers)

            # Alternates between main categories (e.g., HF) and other categories
            for idx in range(0,len(ontopic_matrix)):
                ontopic_row = ontopic_matrix.iloc[idx].tolist()
                other_row = other_matrix.iloc[idx].tolist()
                writer.writerow(ontopic_row)
                writer.writerow(other_row) 

            # Writes the rest of the other categories
            for idx in range(len(ontopic_matrix), len(other_matrix)):
                other_row = other_matrix.iloc[idx].tolist()
                writer.writerow(other_row) 

    elif merge_matrix_option_2:
        '''
        Option 2: Load main and other feature matrices one line at a time and combine them
        This scales better if it is too difficult to load the entire matrix into 
        the working memory
        '''
        with open(combined_feature_matrix_path, 'w') as fout:
            with open(feature_matrix_for_ontopic_categories_path) as fin:
                for idx, line in enumerate(fin):
                    fout.write(line)

            with open(feature_matrix_for_offtopic_category_path) as fin:
                for idx, line in enumerate(fin):
                    if idx == 0: # skip column headers (already included above)
                        continue
                    else:
                        fout.write(line)
    else:
        print('Specify a merging strategy')
        return
    
    with open(f'output/{topic}_feature_matrix_path.txt','w') as fout:
        fout.write(combined_feature_matrix_path)
    
    print('Done!')
    print('Results stored at:')
    print(combined_feature_matrix_path)
    
    
if __name__ == '__main__':
    # Be sure to run the get_pubmed_docs.py first for the topic of interest. This current file
    # relies on the output from that previous API.
    '''
    Example of first PubMed API
    topic = 'hf'
    ! python get_pubmed_docs.py --get_docs_on_pubmed\
                               --get_pmids_via_mesh\
                               --categories f'input/categories_list_of_list_of_tree_numbers_{topic}.json'\
                               --cats_of_pmids f'output/category_of_pmids_{topic}.csv'\
                               --pmid_to_cat f'output/pmid_to_category_{topic}.json'\
                               --ft_mtrx_pth f'output/feature_matrix_{topic}.csv'\

    Example of this PubMed API
    ! python get_offtopic_docs.py --topic f'{topic}' --num_random_pmids 10000 --m1 
    '''
    
    parser = argparse.ArgumentParser(description='PubMed document API, unlabeled docs')
    parser.add_argument('--topic', '-t',
                        type=str, default='hf', 
                        help='the name of the topic you are studying (you should have run the first pubmed API with this topic name, and the file should be available)')
    parser.add_argument('--num_random_pmids', '-n',
                        type=int, default=10000)
    parser.add_argument('--max_pmid', '-max',
                        type=int, default=37000000, help='Maximum PMID ID number')
    parser.add_argument('--min_pmid', '-min',
                        type=int, default=0, help='Minumum PMID ID number')
    parser.add_argument('--get_offtopic_docs', '-off',
                   action='store_true', default=False)    
    parser.add_argument('--get_unlabeled_docs', '-unlab',
                   action='store_true', default=False)
    parser.add_argument('--merge_matrix_option_1', '-m1',
                        action='store_true', default=False, 
                        help='merge random PMIDs and topic-relevant PMIDs via loading whole dataframe into memory')
    parser.add_argument('--merge_matrix_option_2', '-m2',
                        action='store_true', default=False,
                        help='merge random PMIDs and topic-relevant PMIDs in more scalable way')
    args = parser.parse_args()
    topic = args.topic
    num_random_pmids = args.num_random_pmids# determines how many "other" category PMIDs there will be
    min_pmid_number = args.min_pmid         # randomly choose PMIDs above this threshold
    max_pmid_number = args.max_pmid         # randomly choose PMIDs below this threshold
    merge_matrix_option_1 = args.merge_matrix_option_1 
    merge_matrix_option_2 = args.merge_matrix_option_2
    get_labeled_docs = args.get_offtopic_docs     # specify if you want known offtopic docs
    get_unlabeled_docs = args.get_unlabeled_docs  # specify if you want unknown topic docs 
    if get_labeled_docs == True and get_unlabeled_docs == True:
        Exception('Pick either "get_offtopic_docs" or "get_unlabeled_docs"')
    
    # add enough pmids to match the number specified ?
    
    
    get_offtopic_or_unlabeled_pmids(topic, num_random_pmids, max_pmid_number, get_labeled_docs, get_unlabeled_docs)
    get_offtopic_or_unlabeled_documents(num_random_pmids, topic, get_labeled_docs, get_unlabeled_docs)                        
    if get_offtopic_docs:
        merge_ontopic_and_offtopic_feature_matrices(topic, 
                                              merge_matrix_option_1, 
                                              merge_matrix_option_2)    
