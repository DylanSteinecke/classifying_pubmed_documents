# Classifying PubMed Documents
This repository is for the code to classify PubMed documents. PubMed classifies/labels/categorizes documents with MeSH terms in the document metadata. However, about 15% of PubMed is uncategorized. This is typically because there is not enough information provided in the document 

## Document Download

### get_pubmed_docs.py
This script obtains PubMed documents (titles + abstracts) via an API. PubMed documents are chosen based on the topics denoted by MeSH terms. One way to use the API is for the user to submit a list of MeSH-defined categories (e.g., heart failure MeSH tree numbers) in the 'input/categories_list_of_list_of_tree_numbers_{yourtopic}.json'. The other way to use the API is to submit a list of PMIDs to obtain the documents. This second way is used by the get_offtopic_or_unlabeled_docs.py.

Example Usage:
```
topic='hf'
python get_pubmed_docs.py --get_docs_on_pubmed \
                           --get_pmids_via_mesh \
                           --download_mesh_tree \
                           --categories 'input/mesh_terms.json' \
                           --cats_of_pmids "output/category_of_pmids_${topic}.csv" \
                           --pmid_to_cat "output/pmid_to_category_${topic}.json" \
                           --ft_mtrx_pth "output/feature_matrix_${topic}.csv" \
                           --max_num_docs 999999


```

Flags
```
--download_mesh_tree : downloads the MeSH tree. Only needs to be done once.
--categories, -c : the user must create this file, a list of lists of MeSH tree numbers denoting the topics to be studied.
--get_docs_on_pubmed : include this flag if you want to get the PubMed documents
--get_pmids_via_mesh : include this flag if you want to get PubMed documents studying your MeSH terms. Don't include it if you have a predefined set of PMIDs you want.
--ft_mtrx_path : use this to define the output path for the PubMed documents
--max_num_docs : use this to choose the max number of documents you want per topic
```


### get_offtopic_or_unlabeled_docs.py
This script also obtains PubMed documents (titles + abstracts) via an API. However, these documents are a user-chosen number of documents *not* studying your topics of interst as shown. This is used for training the model to discriminate between your topics and other topics it will see when you use the document classifier on unlabeled/uncategorized documents. This relies on the previous API, get_pubmed_docs.py, assuming that it has been run on your topic. This API will use that API's output as input, automatically searching for files created in the first API. 
Example Usage:
```
# Get offtopic PMIDs
topic='hf'
python get_offtopic_or_unlabeled_docs.py --topic $topic \
                                  --num_random_pmids 10000 -m2 \
                                  --min_pmid 37000000 \
                                  --get_offtopic_docs
```

```
# Get unlabeled documents (with PMID greater than 37000000 (more recent))
topic='hf'
python get_offtopic_or_unlabeled_docs.py --topic $topic \
                                  --num_random_pmids 10000 -m2 \
                                  --min_pmid 37000000 \
                                  --get_unlabeled_docs
```

Flags:
```
--topic : specify the topic you are studying. Make sure it matches the name of a topic you have used to run the first API.
--num_random_pmids : specify the number of random PMIDs to use. The script will pick this many random PMIDs, remove any PMIDs that are known to study your topics, and then submit those PMIDs to the get_pubmed_docs.py to retrieve PubMed documents (titles + abstracts) which have been labeled as studying topics other than your topic(s) of interest.
--max_pmid : this is the largest PMID to consider. This is a way to choose PMIDs from a certain date range, because older PMIDs are more likely to have been labeled. Currently, PMIDs prior to 37000000 are more labeled (11/9/23).
--min_pmid : this is the smallest PMID to consider. This is similar to above, but its purpose is for when you want to find more recent and unlabeled documents. For example, set it to 3700000 to find the more recent documents.
-m1 : a behind-the-scenes way to merge the data with dataframes
-m2 : a behind-the-scenes way to merge the data (just pick this one)
```

## Document Classification

### pre_filter_document_classifier.py [In progress]
This applies a preliminary document classifier. (Naive Bayesian Classifier) It is intended to have high on-topic recall and moderate precision, intending to filter out the off-topic documents. By this we mean that the documents studying topics that you have not chosen (e.g., other than 3 types of heart failure) will be filtered out. They will then be fed to the ostensibly high precision PyTorch-based transformer language model classifier which will categorize the documents by topic. This step is important because there is a massive class imbalance for most small sets of topics compared to all other topics (e.g., ~5,000 documents on any type of heart failure, ~38,000,000 documents on all other topics), so by filtering out the majority of the off-topic documents, it can not only expedite but also improve the final classifier. 

### pytorch_document_classifier.py
This is the transformer-based language model that is fine-tuned to classify documents by topic. The training and testing is implemented in PyTorch. Users can specify the model, epochs, etc. 

### run_document_classifier.py
This runs the entire pipeline to download on-topic (get_pubmed_docs.py) and off-topic (get_offtopic_or_unlabeled_docs.py) documents, prompting the user for certain inputs such as the topic name and number of documents. It then runs the document classification (pytorch_document_classifier)
