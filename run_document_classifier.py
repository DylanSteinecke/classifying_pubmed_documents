from torch import cuda
import os
import subprocess


def get_gpu_count():
    try:
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        gpu_names = result.decode("utf-8").strip().split("\n")
        return len(gpu_names)
    except subprocess.CalledProcessError:
        return 0  # No GPUs found or nvidia-smi not installed

    
'''
Pick the GPU to use
'''
os.system('nvidia-smi --query-gpu=utilization.gpu --format=csv')  
num_gpus = get_gpu_count()

while True:
    try:
        gpu_id = int(input("Enter the GPU ID: "))
        break
    except ValueError:
        print("Sorry, I didn't understand that.")
        continue
    if gpu_id > num_gpus-1:
        print("There aren't that many GPUs (there are", num_gpus)
        continue
cuda.set_device(gpu_id)



# Name the topic you are studying
while True:
    try:
        topic = str(input("Enter the topic name (the part before the underscore will be used to find the category file): ")).replace(' ','_')
        break
    except ValueError:
        print("Sorry, I didn't understand that")
        continue


# Ask to redownload the data
while True:
    download = str(input("Do you want to download new data? y/n: "))
    if download in ('y','n'):
        break
    else:
        pass
    
if download == 'y':
    # Define max number of documents in each category
    while True:
        try:
            num_docs_in_each_category = str(int(input("Enter the maximum number of documents in each category: ")))
            break
        except ValueError:
            print("Sorry, I didn't understand that")
            continue

    # Define number of documents in the "other topics" category
    while True:
        try:
            num_docs_in_other = str(int(input("Enter the number of documents in the 'other topics' category: ")))
            break
        except ValueError:
            print("Sorry, I didn't understand that")
            continue

    '''
    Download data
    '''
    # Download topic-relevant documents
    print('*'*50, '\n', 'obtaining topic-relevant documents', '\n', '*'*50, '\n')
    pubmed_doc_cmd = [
        "python", 'pubmed_doc_api.py', 
                        '--get_docs_on_pubmed',
                        '--get_pmids_via_mesh',
                        '--categories',  f'input/categories_list_of_list_of_tree_numbers_{topic.split("_")[0]}.json',
                        '--cats_of_pmids', f'output/category_of_pmids_{topic}.csv',
                        '--pmid_to_cat',   f'output/pmid_to_category_{topic}.json',
                        '--ft_mtrx_pth',   f'output/feature_matrix_{topic}.csv',
                        '--max_num_docs',  num_docs_in_each_category]
    subprocess.run(pubmed_doc_cmd, check=True)

    # Download other topic documents
    print('\n', '*'*50, '\n'+ 'obtaining topic-irrelevant documents', '\n', '*'*50, '\n')
    pubmed_other_cmd = [
        "python", "pubmed_other_docs_api.py",
                        "--topic",  topic,
                        "--num_random_pmids", num_docs_in_other,
                        "-m2"]
    subprocess.run(pubmed_other_cmd, check=True)

    
    
# Load dataset
from pytorch_document_classifier import *

with open(f'output/{topic}_feature_matrix_path.txt','r') as fin:
    feature_matrix_path = fin.readlines()[0].strip()
print('Feature Matrix Path', feature_matrix_path)
train_test_data = prepare_feature_matrix(
                       feature_matrix_path=feature_matrix_path,
                       use_head=False,)
num_labels = len(set(train_test_data['train']['labels']))

'''
Run model
'''
# Classify documents in dataset
DC_cvd = DocumentClassifier(dataset=train_test_data)
DC_cvd.classify_documents(model_name='biolink',
                          epochs=10,
                          num_labels=num_labels,
                          batch_size=16,
                          model_name_suffix=topic,
                          lr=3e-5,
                          logfile=f'output/{topic}_logfile.txt',
                          save_model=True)

