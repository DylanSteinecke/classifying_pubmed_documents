from torch import cuda
import os
import re
import subprocess
from pytorch_document_classifier import *


def get_gpu_utilization():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        gpu_utilizations = [int(utilization) for utilization in result.stdout.strip().split('\n')]
        return gpu_utilizations
    except Exception as e:
        print(f"Error getting GPU utilization: {e}")
        return None

def choose_least_utilized_gpu():
    gpu_utilizations = get_gpu_utilization()

    if gpu_utilizations is not None:
        # Find the GPU index with the lowest utilization
        min_utilization_index = gpu_utilizations.index(min(gpu_utilizations))
        return min_utilization_index
    else:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process input for data download')
    parser.add_argument('--topic', type=str, help='Topic name')                                                                            # Must specify
    parser.add_argument('--download', action='store_true', default=False, help='Download new data')
    parser.add_argument('--num_ontopic_topic_docs', type=int, help='Max number of documents in each ontopic category')                     # Must specify
    parser.add_argument('--num_offtopic_docs', type=int, help='Number of offtopic documents')                                              # Must specify
    parser.add_argument('--num_unlabeled_docs', type=int, help='Number of unlabeled documents')                                            # Must specify
    parser.add_argument('--train_stage_one_classifier', type='store_true', default=False, help='Train the language model')
    parser.add_argument('--train_stage_two_classifier', type='store_true', default=False, help='Train the naive Bayes classifier')
    parser.add_argument('--run_stage_one_classifier', type='store_true', default=False, help='Run the naive Bayes classifier')
    parser.add_argument('--run_stage_two_classifier', type='store_true', default=False, help='Run the language model classifier')
    parser.add_argument('--model_name', type=str, help='Name of the HuggingFace model', default='biolink')
    parser.add_argument('--use_original_for_stage_two_training', '-use_entire', action='store_true', default=False)             # Pick this or...
    parser.add_argument('--use_stage_one_predicted_positives_for_stage_two_training', '-use_s1_preds', action='store_true', default=False) # ...this.
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    topic = args.topic
    download = args.download
    num_ontopic_docs = args.num_ontopic_topic_docs
    num_offtopic_docs = args.num_offtopic_docs
    num_unlabeled_docs = args.num_unlabeled_docs

    ### Pick the GPU to use ###
    os.system('nvidia-smi --query-gpu=utilization.gpu --format=csv')  
    chosen_gpu_id = choose_least_utilized_gpu()
    if chosen_gpu_id:
        print(f"Chosen GPU: {chosen_gpu}")
    else:
        Exception("Failed to determine GPU utilization.")
    cuda.set_device(chosen_gpu_id)

    
    ########################
    ## Document Download  ##
    ########################
    if download:        
        ### Download ontopic documents ###
        print('*'*50, '\n', 'obtaining topic-relevant documents', '\n', '*'*50, '\n')
        pubmed_doc_cmd = [
            "python", 'get_pubmed_docs.py', 
                            '--get_docs_on_pubmed',
                            '--get_pmids_via_mesh',
                            '--categories',  f'input/categories_list_of_list_of_tree_numbers_{topic.split("_")[0]}.json',
                            '--cats_of_pmids', f'output/category_of_pmids_{topic}.csv',
                            '--pmid_to_cat',   f'output/pmid_to_category_{topic}.json',
                            '--ft_mtrx_pth',   f'output/feature_matrix_{topic}.csv',
                            '--max_num_docs',  num_ontopic_topic_docs]
        subprocess.run(pubmed_doc_cmd, check=True)
    
        ### Download offtopic documents ###
        print('\n', '*'*50, '\n'+ 'obtaining topic-irrelevant documents', '\n', '*'*50, '\n')
        pubmed_offtopic_cmd = [
            "python", "pubmed_offtopic_or_unlabeled_docs_api.py",
                            "--topic",  topic,
                            "--num_of_pmids", num_offtopic_docs,
                            "--get_offtopic_docs",
                            "--max_pmid", 37000000,
                            "-m2"]
        subprocess.run(pubmed_offtopic_cmd, check=True)
    
        ### Download unlabeled documents ###
        print('\n', '*'*50, '\n'+ 'obtaining unlabeled documents', '\n', '*'*50, '\n')
        pubmed_unlabeled_cmd = [
            "python", "pubmed_offtopic_or_unlabeled_docs_api.py",
                            "--topic",  topic,
                            "--num_of_pmids", num_offtopic_docs,
                            "--get_unlabeled_docs",
                            "--min_pmid", 37000000,]
        subprocess.run(pubmed_unlabeled_cmd, check=True)


    ##########################
    ## Stage One Classifier ##
    ##########################
    # Find the path to the feature matrix
    if args.train_stage_one_classifier:
        # Load Training Data    
        path_to_feature_matrix_path = f'output/{topic}_original_feature_matrix_path.txt' # on+offtopic w/ ground truth labels
    elif args.run_stage_one_classifier:
        # Load unlabeled data 
        path_to_feature_matrix_path = f'output/{topic}_unlabeled_{num_offtopic_docs}_docs_feature_matrix_path.txt' # on(?)+offtopic w/ no labels
    with open(path_to_feature_matrix_path,'r') as fin:
        feature_matrix_path = fin.readlines()[0].strip()

    # Simha: use feature_matrix_path to read the feature matrix in (this is the path to the feature/document matrix)
    
    # Run Naive Bayes classifier 
    #print('\n', '*'*50, '\n'+ 'Running Naive Bayes classifier', '\n', '*'*50, '\n')
    #naive_bayes_cmd = [
    #    "python", "___.py",
    #                    "--topic",  topic,
    #                  ]
    #subprocess.run(pubmed_unlabeled_cmd, check=True)


    ###########################
    ## Stage Two Classifier  ##
    ###########################
    # Find the path to the feature matrix
    if args.train_stage_two_classifier:
        # Load Training Data    
        if args.use_original_for_stage_two_training:
            path_to_feature_matrix_path = f'output/{topic}_original_feature_matrix_path.txt'
        elif args.use_stage_one_predicted_positives_for_stage_two_training:
            path_to_feature_matrix_path = f'output/{topic}_stage_one_labeled_positive_feature_matrix_path.txt' # ground truth labels
    elif args.run_stage_two_classifier:
        if args.run_stage_one_classifier:    
            # Load unlabeled data predicted to be ontopic by stage one
            path_to_feature_matrix_path = f'output/{topic}_stage_one_predicted_postive_feature_matrix_path.txt'    # stage 1 labels
        else:
            # Load unlabeled data 
            path_to_feature_matrix_path = f'output/{topic}_unlabeled_{num_offtopic_docs}_docs_feature_matrix_path.txt' # on(?)+offtopic w/ no labels

    with open(path_to_feature_matrix_path,'r') as fin:
        feature_matrix_path = fin.readlines()[0].strip()

    # Load the feature matrix
    train_test_data = prepare_feature_matrix(feature_matrix_path)
    num_labels = len(set(train_test_data['train']['labels']))
    
    # Run model 
    DC_cvd = DocumentClassifier(dataset=train_test_data)
    DC_cvd.classify_documents(model_name=args.model_name',
                              epochs=args.epochs,
                              num_labels=num_labels,
                              batch_size=16,
                              model_name_suffix=topic,
                              lr=3e-5,
                              logfile=f'output/{topic}_logfile.txt',
                              save_model=True)
