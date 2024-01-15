from torch import cuda
import os
import re
import subprocess
import argparse
import json
from pytorch_document_classifier import *


def get_gpu_utilization():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        gpu_utilizations = [int(utilization) for utilization in result.stdout.strip().split('\n')]
        return gpu_utilizations
    except Exception as e:
        print(f'Error getting GPU utilization: {e}')
        return None

def choose_least_utilized_gpu():
    gpu_utilizations = get_gpu_utilization()

    if gpu_utilizations is not None:
        # Find the GPU index with the lowest utilization
        min_utilized_gpu = min(gpu_utilizations)
        print(f'GPU is at {min_utilized_gpu} capacity')
        if min_utilized_gpu > 0:
            raise Exception('GPU might be too busy')
        min_utilization_index = gpu_utilizations.index(min_utilized_gpu)
        return min_utilization_index
    else:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input for data download')
    parser.add_argument('--topic', 
                        type=str, help='Topic name') # Must specify
    parser.add_argument('--download_docs', '-d',
                        action='store_true', default=False, help='Download new data')
    parser.add_argument('--num_ontopic_topic_docs', type=str, # Must specify
                        help='Max number of documents in each ontopic category')
    parser.add_argument('--num_offtopic_docs', type=int,
                        help='Number of offtopic documents') # Must specify
    parser.add_argument('--num_unlabeled_docs', type=int, 
                        help='Number of unlabeled documents') # Must specify
    parser.add_argument('--train_stage_one_classifier', action='store_true', 
                        default=False, help='Train the Naive Bayes classifier')
    parser.add_argument('--train_stage_two_classifier', action='store_true', 
                        default=False, help='Train the language model')
    parser.add_argument('--train_stage_three_classifier', action='store_true', 
                        default=False, help='Train the language model')
    parser.add_argument('--run_stage_one_classifier', action='store_true', 
                        default=False, help='Run the naive Bayes classifier')
    parser.add_argument('--run_stage_two_classifier', action='store_true', 
                        default=False, help='Run the language model classifier')
    parser.add_argument('--model_name', type=str, 
                        help='Name of the HuggingFace model', default='biolink')
    parser.add_argument('--use_original_for_stage_two_training', '-use_entire', 
                        action='store_true', default=False)  # Pick this or...
    parser.add_argument('--use_stage_one_predictions', '-use_s1_preds', 
                        action='store_true', default=False) # ...this.
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--max_pmid', '-max',
                        type=int, default=38000000, 
                        help='Maximum PMID ID number')
    parser.add_argument('--min_pmid', '-min',
                        type=int, default=0, 
                        help='Minumum PMID ID number of offtopic documents')
    parser.add_argument('--min_pmid_unlabeled', '-min_un',
                        type=int, default=37000000,
                        help='Minumum PMID ID number of unlabeled documents')
    parser.add_argument('--batch_size', '-b_s',
                        type=int, default=16, help='batch size for stage 2 LM classifier')
    parser.add_argument('--test_size_stage_1', '-ts1', 
                        type=float, default=0.50, help='Test split ratio for stage 1, NBC')
    parser.add_argument('--test_size_stage_2', '-ts2', 
                        type=float, default=0.50, help='Test split ratio for stage 2, LM')
    args = parser.parse_args()

    topic = args.topic
    download = args.download_docs
    num_ontopic_docs = args.num_ontopic_topic_docs
    num_offtopic_docs = args.num_offtopic_docs
    num_unlabeled_docs = args.num_unlabeled_docs
    min_pmid = args.min_pmid
    max_pmid = args.max_pmid
    min_pmid_unlabeled = args.min_pmid_unlabeled
    batch_size = args.batch_size
    test_size_stage_1 = args.test_size_stage_1
    test_size_stage_2 = args.test_size_stage_2
    

    ########################
    ## Document Download  ##
    ########################
    if download:        
        ### Download ontopic documents ###
        print('*'*50, '\nobtaining labeled on-topic documents\n', '*'*50, '\n')
        categories_path = f'input/{topic.split("_")[0]}_tree_numbers.json'
        cats_of_pmid_path = f'output/{topic}/category_of_pmids_{topic}.csv'
        pmid_to_cat_path = f'output/{topic}/pmid_to_category_{topic}.json'
        ft_mtrx_path = f'output/{topic}/feature_matrix_{topic}.csv'
        pubmed_doc_cmd = [
            'python', 'get_pubmed_docs.py', 
                            '--topic', topic,
                            '--download_mesh_tree',
                            '--get_docs_on_pubmed',
                            '--get_pmids_via_mesh',
                            '--categories', categories_path,
                            '--cats_of_pmids', cats_of_pmid_path,
                            '--pmid_to_cat', pmid_to_cat_path,
                            '--ft_mtrx_pth', ft_mtrx_path,
                            '--max_num_docs', num_ontopic_docs]
        subprocess.run(pubmed_doc_cmd, check=True)
    
        ### Download offtopic documents ###
        print('\n','*'*50,'\n'+'obtaining labeled off-topic documents','\n','*'*50,'\n')
        pubmed_offtopic_cmd = [
            'python', 'get_offtopic_or_unlabeled_docs.py',
                           '--topic',  topic,
                            '--num_of_pmids', str(num_offtopic_docs),
                            '--get_offtopic_docs',
                            '--min_pmid', str(min_pmid),
                            '--max_pmid', str(max_pmid),
                            '-m2',]
        subprocess.run(pubmed_offtopic_cmd, check=True)
    
        ### Download unlabeled documents ###
        print('\n', '*'*50, '\n'+ 'obtaining unlabeled documents', '\n', '*'*50, '\n')
        pubmed_unlabeled_cmd = [
            'python', 'get_offtopic_or_unlabeled_docs.py',
                            '--topic',  topic,
                            '--num_of_pmids', str(num_unlabeled_docs),
                            '--get_unlabeled_docs',
                            '--min_pmid', '37000000',]
        subprocess.run(pubmed_unlabeled_cmd, check=True)




    ##########################
    ## Stage One Classifier ##
    ##########################
    off_topic_class_num = len(json.load(open(f'input/{topic}_tree_numbers.json')))
    
    # Train Model
    if args.train_stage_one_classifier:
        
        # Load Training Data    
        path = f'output/{topic}/{topic}_original_feature_matrix_path.txt'
        path_to_labeled_feature_matrix_path = path
        with open(path_to_labeled_feature_matrix_path,'r') as fin:
            labeled_feature_matrix_path = fin.readlines()[0].strip()
        
            
        # Train Naive Bayes classifier 
        print('\n', '*'*50, '\n'+ 'Running Naive Bayes classifier', '\n', '*'*50, '\n')
        naive_bayes_cmd = [
            'python3', './NBC/NBC.py',
                            '--run_mode', 'train_test',
                            '--input_path', labeled_feature_matrix_path,
                            '--off_topic_class', str(off_topic_class_num),
                            '--topic',  topic,
                            '--out_path', f'output/{topic}',
                            '--test_size_stage_one', str(test_size_stage_1),
                        ]
        subprocess.run(naive_bayes_cmd, check=True)

    # Inference Time / Deploy Model
    elif args.run_stage_one_classifier:
        # Load unlabeled data 
        path = f'output/{topic}/{num_unlabeled_docs}_unlabeled_docs_feature_matrix_path.csv'
        path_to_unlabeled_feature_matrix_path = path
        with open(path_to_unlabeled_feature_matrix_path,'r') as fin:
            unlabeled_feature_matrix_path = fin.readlines()[0].strip()            
        
        naive_bayes_cmd = [
            'python', './NBC/NBC.py',
                            '--run_mode', 'predict_unlabeled ',
                            '--topic',  topic,
                            '--unlabeled_docs_path', unlabeled_feature_matrix_path,
                            '--out_path', f'output/{topic}',
                        ]
        subprocess.run(naive_bayes_cmd, check=True)

    

    ##########################
    ## Stage Two Classifier ##
    ##########################
    
    ### Pick the GPU to use ###
    # Check for CUDA capability 
    if cuda.is_available():
        print('CUDA is available. Using GPU.')
        chosen_gpu_id = choose_least_utilized_gpu()
        if chosen_gpu_id is not None:
            print(f'Chosen GPU: {chosen_gpu_id}')
            cuda.set_device(chosen_gpu_id)
    else:
        print('!!!! NOTE: CUDA is not available. Using CPU !!!!')
    

    # Train Model
    direct = f'output/{topic}'
    if args.train_stage_two_classifier:
        mode = 'train'
        stage_2 = True
        if args.use_original_for_stage_two_training: # Original PubMed-downloaded data
            path = f'{direct}/{topic}_original_feature_matrix_path.txt' 
            path_to_feature_matrix_path = path
        elif args.use_stage_one_predictions:  # NB-labeled on-topic (ground truth labels) 
            feature_matrix_path = f'{direct}/NBC_test_docs_pred_ontopic_{topic}.csv' 

    # Inference Time / Deploy Model
    elif args.run_stage_two_classifier:
        mode = 'inference'
        stage_2 = True
        # Classify the NB-predictions labeled by Naive Bayes 
        if args.use_stage_one_predictions: 
            path = f'{direct}/{topic}_stage_one_predicted_postive_feature_matrix_path.txt' 
            path_to_feature_matrix_path = path
        # Classify the unpredicted original dataset
        else: 
            path = f'{direct}/{num_unlabeled_docs}_unlabeled_docs_feature_matrix_path.csv'
            path_to_feature_matrix_path = path
    else:
        stage_2 = False
        #exit('Note: Stage Two Classifier Not Run')
    
    
    if stage_2:
        
        # Load the feature matrix
        try:
            with open(path_to_feature_matrix_path,'r') as fin:
                feature_matrix_path = fin.readlines()[0].strip()
        except:
            if feature_matrix_path:
                pass
            else:
                raise Exception('No feature matrix / training set for stage 2')

        # Process data
        if feature_matrix_path:
            train_test_data = prepare_feature_matrix(feature_matrix_path, test_size_stage_2)
            num_labels = len(set(train_test_data['train']['labels']))
            assert off_topic_class_num == num_labels-1

            # Run model 
            DC_2 = DocumentClassifier(dataset=train_test_data, topic=topic, stage_num='2')
            DC_2.classify_documents(model_name=args.model_name,
                                   epochs=args.epochs,
                                   num_labels=num_labels,
                                   batch_size=args.batch_size,
                                   model_name_suffix=topic+'_'+mode,
                                   lr=3e-5,
                                   save_model=True)

    
    ############################
    ## Stage Three Classifier ##
    ############################

    # Train Model
    direct = f'output/{topic}'
    if args.train_stage_three_classifier:
        feature_matrix_path = f'{direct}/stage_2_test_docs_pred_ontopic_{topic}.csv' 
    else:
        exit('Note: Stage Three Classifier Not Run')
        
    train_test_data = prepare_feature_matrix(feature_matrix_path)
    num_labels = len(set(train_test_data['train']['labels']))
    assert off_topic_class_num == num_labels-1
    
    # Run model 
    mode = 'train'
    DC_3 = DocumentClassifier(dataset=train_test_data, topic=topic, stage_num='3')
    DC_3.classify_documents(model_name=args.model_name,
                              epochs=args.epochs,
                              num_labels=num_labels,
                              batch_size=args.batch_size,
                              model_name_suffix=topic+'_'+mode+'_stage_3',
                              lr=3e-5,
                              save_model=True)
