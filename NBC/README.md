**Overview**

NBC.py is a Python script implementing a Naive Bayes Classifier (NBC) tailored for categorizing PubMed documents. It is designed to classify documents into specified categories based on their content, particularly useful in biomedical literature analysis.

**Prerequisites**

Before running the script, ensure you have Python installed on your system. The script has been tested with Python 3.8+. Additionally, the required Python packages can be installed via the requirements.txt file.

**Installation**
First, clone the repository or download the NBC.py script and the requirements.txt file. Install the necessary dependencies by running:

```pip install -r requirements.txt```

**Input Data Format**

To train the model, the script expects a CSV file as input with specific columns (it is okay to have other columns in addition to this) :

	title: Title of the PubMed document.

	abstract: Abstract of the document.

	topic_labels: Integer labels indicating the document's category. (e.g. 0,1,2,3) 
 
IMPORTANT: 

	The NBC classifier is structured as a binary classifier, effectively categorizing documents into two distinct groups. All documents not tagged as "off topic" are consolidated under a single label. For example, documents labeled as 'diastolic heart failure' and 'systolic heart failure', labeled as 1 and 2, would both be reclassified as label 1. Conversely, any document identified as 'off topic' would be uniformly reassigned the label 0. This process ensures a clear binary division in the dataset, with one label for relevant topics and another for off-topic content.

To make predictions on unlabeled abstracts, the scrip expects a CSV file as input with one specific column (it is okay to have other columns in addition to this) : 

	abstract: Abstract of the document.

**Command-Line Arguments**

The script accepts several command-line arguments to control its behavior:

	--num_classes (default=2): The number of classes for classification.
	--off_topic_class: Specifies the class representing documents not in the disease of interest.
	--input_path: The path to the input CSV file containing labeled data.
	--out_path (default='./output'): The directory where output files will be saved.
	--unlabeled_docs_path: Path for the CSV file containing unlabeled documents (required for predict_unlabeled mode).
	--run_mode (required): The mode to run the script in, either train_test or predict_unlabeled.
 	
  	Run Modes:
    		train_test: This mode trains the Naive Bayes Classifier on a provided dataset and evaluates its performance on a test split.
    		predict_unlabeled: In this mode, the Classifier is first trained and evaluated on the train test split and then the model is used to predict the categories of unlabeled documents.
 
**EXAMPLE**

TRAIN TEST MODE 

```
python ./NBC/NBC.py \
  --run_mode train_test \
  --input_path /content/feature_matrix_7810_non_hf_4426_hf.csv \
  --off_topic_class 5
```

PREDICT UNlABELED MODE 

```
python ./NBC/NBC.py \
  --run_mode predict_unlabeled \
  --input_path /content/feature_matrix_7810_non_hf_4426_hf.csv \
  --off_topic_class 5 \
  --unlabeled_docs_path /content/unlabeled.csv
```

**OUTPUT**
The script generates various outputs based on the chosen mode, including:

Word counts and log likelihoods for each class.

Classification reports and confusion matrices for the train_test mode.

A CSV file of the documents in the test split that were predicted to be on topic. 

Predicted categories for unlabeled documents in predict_unlabeled mode.

All outputs are saved in the directory specified by --out_path.



