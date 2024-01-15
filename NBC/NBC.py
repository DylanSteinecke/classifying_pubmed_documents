import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import argparse

# Download required NLTK resources
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

class NBC:
    def __init__(self, num_classes, off_topic_class,
                 input_path, out_path, topic, test_size=0.30):
        self.description = 'Naive Bayes Classifier for PubMed Documents'
        
        self.topic = topic # Topic being studied
        self.num_classes = num_classes
        self.num_docs_per_class = None
        self.off_topic_class =  off_topic_class # Num of class not of interest
        self.labeled_pubmed = 0.85*37000000 # Hardcoded approximate number
        self.input_path = input_path

        # Train test splits
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_size = test_size
        self.y_pred = []
        

        self.lemmatize = WordNetLemmatizer().lemmatize
        self.stop_words = set(stopwords.words('english'))
        self.vocabulary = set()
        self.class_word_counts = defaultdict(Counter)
        self.loglikelihoods = {label: {} for label in range(num_classes)}

        # Initialize list that stores indices of docs predicted to be heart failure
        self.hf_pred_docs = []
        
        # Evaluation metrics
        self.accuracy_score = None
        self.precision = None
        self.recall = None
        self.f1 = None
        
        # Path for output directory
        self.directory = out_path
        
        # Initialize variable to save original dataframe with all labels
        self.orig_df = None

        
    def prior_c(self):
        '''
        Calculate P(class) terms based on num docs in on topic and off topic class (1 and 0 
        respectively) since we take a random sample of all off topic documents. I calculate 
        total # off topic docs as total labeled docs minus the # of docs in on topic class 
        '''
        # Call the prepare_feature_matrix method to get the training and test data
        if self.df is None:  # Check if the dataset has already been prepared
            self.prepare_feature_matrix()
        p_zero = (int(self.labeled_pubmed - self.num_docs_per_class[1]))/self.labeled_pubmed
        p_one = self.num_docs_per_class[1]/self.labeled_pubmed
        prior_c = {0: p_zero,
                   1: p_one}
        return prior_c



    def prepare_feature_matrix(self):
        '''
        Remove rows with documents lacking abstracts and with multiple topics of interest.
        Then, convert the problem into binary classification. Then, split into train/test.
        '''
        df = pd.read_csv(self.input_path)

        # Remove documents without abstracts and with multiple mesh labels
        df = df[~df.labels.astype(str).str.contains(',')]
        df = df.dropna(subset=['abstract'])

        # Change labels column to integers
        df.loc[df['labels'].apply(lambda x: isinstance(x, int))]   

        # Concatenate titles and abstracts
        df.loc[:, 'abstract'] = df['title']+' '+df['abstract']
        df = df.drop('title', axis=1)

        ## Combine on_topic into one group --> label 0 (e.g., systolic and diastolic HF)
        ## Replace off_topic label with --> 1 (e.g., non heart failure)
        df.loc[:, 'labels'] = df['labels'].apply(
                                    lambda x: 1 if x == self.off_topic_class else 0)
        self.off_topic_class = 1

        # Get number of documents per class
        self.num_docs_per_class = dict(df.labels.value_counts())

        ## Split into train and test datasets
        X = df['abstract']
        y = df['labels']
        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(X, y, test_size=self.test_size, stratify=y, random_state=42))
        self.df = df
        

    def preprocess(self, abstract):
        '''
        Tokenize the sentences. Then, lemmatize them, removing punctuation, 
        non-alphabetic characters, stop words.
        '''
        tokens = nltk.word_tokenize(abstract.lower()) # Tokenize sentences
        tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]
        preprocessed_text = [self.lemmatize(word) for word in tokens]
        return preprocessed_text


    def train(self):
        '''
        Train the Naive Bayes classifier for binary classification of 
        on-topic vs. off-topic documents.
        '''
        # Call the prepare_feature_matrix method to get the training and test data
        if self.df is None:  # Check if the dataset has already been prepared
            self.prepare_feature_matrix()

        #### Get word counts ####
        for abstract, label in zip(self.X_train, self.y_train):
        #get tokens from the abstract and filter the words (see preprocess())
            tokens = self.preprocess(abstract)
            self.vocabulary.update(tokens)  # Build the vocabulary 
            #add counts for each token in abstract for the given label(class)
            self.class_word_counts[label].update(tokens)   # Dylan, question

        #### Output the word counts in each class into a txt file #####
        output_filename = f'NBC_class_word_counts_{self.topic}.txt'
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        file_path = os.path.join(self.directory, output_filename)

        with open(file_path, 'w') as fout:
            for label, word_counts in self.class_word_counts.items():
                line = f"Class {label} word counts: {word_counts}\n\n"
                fout.write(line)
            fout.close()

        #### Loglikelihoods from training data ####
        # For each class, calculate log likelihood for each word given the clas
        for label, word_counts in self.class_word_counts.items():
            total_words_in_class = sum(word_counts.values())
            #we need word counts for each word in the vocabulary for each class
            #words that have a count of 0 in a class are adjusted due to the add 1 smoothing
            for word in self.vocabulary:
                #compute log likelihood for the word given the class
                word_count_in_class = word_counts.get(word, 0)
                likelihood = (word_count_in_class + 1)/(total_words_in_class + len(self.vocabulary)) # Dylan, question
                self.loglikelihoods[label][word] = np.log(likelihood)

        # save the loglikelihoods to a file similar to what you did for word_counts
        output_filename_likelihood = f'NBC_loglikelihoods_{self.topic}.txt'
        file_path_likelihood = os.path.join(self.directory, output_filename_likelihood)
        with open(file_path_likelihood, 'w') as fout:
            for label, likelihoods in self.loglikelihoods.items():
                line = f"Class {label} log likelihoods: {likelihoods}\n\n"
                fout.write(line)
            fout.close()

        print(f'Log likelihoods for each class saved to {file_path_likelihood}')
        print(f'Word counts for each class saved to {file_path}')

              
    def predict_test_split(self):
        '''
        Predict on training and testing data that has ground truth. 
        Compute evaluation metrics. 
        '''
        for idx, test_doc in zip(self.X_test.index, list(self.X_test)) :
            p = defaultdict(float)
            test_doc = self.preprocess(test_doc)
            for label in self.prior_c().keys():
                p[label] = self.prior_c()[label]
                for word in test_doc:
                    if word in self.vocabulary:
                        p[label] += self.loglikelihoods[label][word]
            self.y_pred.append(max(p, key=p.get))
            if max(p, key=p.get) != 0:
                self.hf_pred_docs.append(idx)

        # Output test docs that were predicted to be on topic (csv)
        on_topic_pred_path = f'{self.directory}/NBC_test_docs_pred_ontopic_{self.topic}.csv'
        on_topic_pred = self.df[self.df.index.isin(self.hf_pred_docs)]
        on_topic_pred.to_csv(on_topic_pred_path)

        # Compute and output the evaluation metrics (txt)
        output_filename = 'test_prediction_metrics.txt'
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        file_path_metrics = os.path.join(self.directory, output_filename)

        with open(file_path_metrics, 'w') as file:
            # Accuracy 
            print('self.y_test', self.y_test, '\n\n\nself.y_pred', self.y_pred)
            print('len(self.y_test)', len(self.y_test), 
                  '\n\n\nlen(self.y_pred)', len(self.y_pred))
            self.accuracy_score = accuracy_score(self.y_test, self.y_pred)
            file.write(f'Accuracy Score: {self.accuracy_score}\n\n')

            # Precision
            self.precision = precision_score(self.y_test, self.y_pred, average='macro') 
            file.write(f'Precision: {self.precision}\n\n')

            # Recall
            self.recall = recall_score(self.y_test, self.y_pred, average='macro')
            file.write(f'Recall: {self.recall}\n\n')

            # F1 score
            self.f1 = f1_score(self.y_test, self.y_pred, average='macro')
            file.write(f'F1 Score: {self.f1}\n\n')

            # Classification report
            report = classification_report(self.y_test, self.y_pred)
            file.write(report)
            print('Naive Bayes Classification Metrics Report:\n', report, '\n')
            print(f'Precision: {self.precision}')
            print(f'Recall: {self.recall}')
            print(f'Accuracy Score: {self.accuracy_score}')
            file.close()

            # Confusion matrix
            cm = confusion_matrix(self.y_test, self.y_pred)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm).plot(ax=ax)
            plt.savefig(f'{self.directory}/NBC_confusion_matrix_{self.topic}.png')
            plt.close(fig)
            

    def predict_unlabeled(self, unlabeled_docs):
        '''
        For predicting at inference time, in settings when there is no 
        ground truth for the examples/documents.
        '''
        df = pd.read_csv(unlabeled_docs)
        pred_class = []
        for doc in df.abstract:
            p = defaultdict(float)
            doc = self.preprocess(doc)
            for label in range(2):
                p[label] = self.prior_c()[label]
                for word in doc:
                    if word in self.vocabulary:
                        p[label] += self.loglikelihoods[label][word]
            pred_class.append(max(p, key=p.get))
        df['NBC_Predicted_Class'] = pred_class

        #Save df as csv file 
        output_file = f'NBC_unlabeled_docs_prediction_{self.topic}.csv'
        file_path = os.path.join(self.directory, output_file)
        df.to_csv(file_path)

        
def main():
    parser = argparse.ArgumentParser(
                  description='Naive Bayes classifier for PubMed documents')
    parser.add_argument('--topic', type=str, 
                        help='Topic name') 
    parser.add_argument('--num_classes', type=int, default = 2, 
                        help='Number of classes')
    parser.add_argument('--off_topic_class', type=int, 
                        help='Class that contains documents not in disease of interest')
    parser.add_argument('--input_path', type=str, 
                        help='Path to input CSV file')
    parser.add_argument('--out_path', type=str, 
                        help='Path for output directory')
    parser.add_argument('--unlabeled_docs_path', type=str, 
                        help='Path for unlabeled documents CSV w/ title & abstract cols')
    parser.add_argument('--run_mode', type=str, choices=['train_test', 'predict_unlabeled'],
                        required=True, help='Either training/testing or inference time')
    parser.add_argument('--test_size_stage_one', '-ts1', 
                        type=float, default=0.50, help='Test split ratio for stage 1, NBC')
    args = parser.parse_args()

    # Instantiate Naive Bayes object
    nbc = NBC(args.num_classes, args.off_topic_class, 
              args.input_path, args.out_path, args.topic,
              args.test_size_stage_one)

    if args.run_mode == 'train_test':
        nbc.prepare_feature_matrix()
        nbc.prior_c()
        nbc.train()
        nbc.predict_test_split()

    elif args.run_mode == 'predict_unlabeled':
        if not args.unlabeled_docs_path:
            raise ValueError("Specify the unlabeled documents path for 'predict_unlabeled'")
        nbc.predict_unlabeled(args.unlabeled_docs_path)

if __name__ == "__main__":
    main()
