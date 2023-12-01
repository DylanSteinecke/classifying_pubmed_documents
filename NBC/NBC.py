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
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

class NBC:
    def __init__(self, num_classes, off_topic_class , input_path, out_path = './output'):
      self.description = 'Naive Bayes Classifier for PubMed Documents'
      self.num_classes = num_classes
      #initialize variable to store # of documents in each class
      self.num_docs_per_class = None
      #class that contains documents not in disease of interest
      self.off_topic_class =  off_topic_class
      #approximation of number of labeled pubmed documents
      self.labeled_pubmed = 0.85*37000000
      self.input_path = input_path
      # Initialize lemmatizer and stop words list
      self.lemmatizer = WordNetLemmatizer()
      self.stop_words = set(stopwords.words('english'))
      # Initialize the variables to hold train and test splits
      self.df = None
      self.X_train = None
      self.X_test = None
      self.y_train = None
      self.y_test = None
      # initialize vocab set and class-wise word counts dict
      self.vocabulary = set()
      self.class_word_counts = defaultdict(Counter)
      # Initialize a dictionary to store log likelihoods for each word given a class
      self.loglikelihoods = {label: {} for label in range(num_classes)}
      #intialize for predictions
      self.y_pred = []
      #initialize list that stores indices of docs predicted to be heart failure
      self.hf_pred_docs = []
      #initialize variables to store metrics
      self.accuracy_score = None
      self.precision = None
      self.recall = None
      self.f1 = None
      #path for output directory
      self.directory = out_path
      #initialize variable to save original dataframe with all labels
      self.orig_df = None

    def prior_c(self):
      # Call the prepare_feature_matrix method to get the training and test data
      if self.df is None:  # Check if the dataset has already been prepared
          self.prepare_feature_matrix()

      ## Calculate P(c) terms based on num docs in on topic and off topic class (1 and 0 respectively)
      ## since we take a random sample of all off topic documents...
      ## I calculate total # off topic docs as total labeled docs minus the # of docs in on topic class
      prior_c = {0: (int(self.labeled_pubmed - self.num_docs_per_class[1]))/self.labeled_pubmed,
                 1: self.num_docs_per_class[1]/self.labeled_pubmed}
      return prior_c

    def prepare_feature_matrix(self):
      mat = pd.read_csv(self.input_path)

      # Filter out documents without abstracts and filter out documents with multiple mesh labels
      mat_filt = (mat[~mat.topic_labels.str.contains(',')]
                  .dropna(subset=['abstract']))

      # Change labels column to integers
      mat_filt['labels'] = mat_filt['topic_labels'].astype(int)

      # Concatenate titles and abstracts
      mat_filt['abstract'] = mat_filt['title'] + ' ' + mat_filt['abstract']
      self.df = mat_filt

      ## Combine documents that are in disease of interest (on topic) into one group --> 1
      ## Replace documents that are off_toic with label --> 0
      ## E.g. combine systolic and diastolic HF docs into one class to compare to non HF docs
      self.df['labels'] = self.df['labels'].apply(
          lambda x: 0 if x == self.off_topic_class else 1
      )
      self.off_topic_class = 0

      # extract number of documents in each class
      self.num_docs_per_class = {idx : count for count, idx in zip(self.df.labels.value_counts(),self.df.labels.value_counts().index)}

      ## Split the dataset manually using scikit-learn
      ### Store the data splits as instance attributes
      self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
          mat_filt['abstract'], mat_filt['labels'], test_size=0.2, stratify=mat_filt['labels'], random_state=42 )

    def preprocess(self, abstract):
      #tokenize sentence
      tokens = nltk.word_tokenize(abstract.lower())

      # Removes punctuation and non-alphabetic characters
      # Remove stop words
      # Lemmatizes words
      return [self.lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in self.stop_words]


    def train(self):
      # Call the prepare_feature_matrix method to get the training and test data
      if self.df is None:  # Check if the dataset has already been prepared
          self.prepare_feature_matrix()

      #### Get word counts ####
      for abstract, label in zip(self.X_train, self.y_train):
        #get tokens from the abstract and filter the words as defined in the preprocess function
        tokens = self.preprocess(abstract)
        #build the vocabulary
        self.vocabulary.update(tokens)
        #add counts for each token in abstract for the given label(class)
        self.class_word_counts[label].update(tokens)

      #### Output the word counts in each class into a txt file #####
      output_filename = 'class_word_counts.txt'
      file_path = os.path.join(self.directory, output_filename)

      #create directory
      if not os.path.exists(self.directory):
        os.mkdir(self.directory)

      # Open the file with write permission and write the contents
      with open(file_path, 'w') as file:
        for label, word_counts in self.class_word_counts.items():
            #Create a string to represent the current label and word counts
            line = f"Class {label} word counts: {word_counts}\n\n"
            # Write this line to the file
            file.write(line)
        file.close()

      #### Loglikelihoods from training data ####
      # For each class, calculate log likelihood for each word
      for label, word_counts in self.class_word_counts.items():
        total_words_in_class = sum(word_counts.values())
        #we need word counts for each word in the vocabulary for each class
        #words that have a count of 0 in a class are adjusted due to the add 1 smoothing
        for word in self.vocabulary:
          #compute log likelihood for the word given the class
          word_count_in_class = word_counts.get(word, 0)
          likelihood = (word_count_in_class + 1)/ (total_words_in_class + len(self.vocabulary))
          self.loglikelihoods[label][word] = np.log(likelihood)

      # save the loglikelihoods to a file similar to what you did for word_counts
      output_filename_likelihood = 'loglikelihoods.txt'
      file_path_likelihood = os.path.join(self.directory, output_filename_likelihood)
      with open(file_path_likelihood, 'w') as file:
          for label, likelihoods in self.loglikelihoods.items():
              line = f"Class {label} log likelihoods: {likelihoods}\n\n"
              file.write(line)
          file.close()

      print(f"Log likelihoods for each class saved to {file_path_likelihood}")
      print(f"Word counts for each class saved to {file_path}")

    def predict_test_split(self):
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

      ### Output csv of test docs that were predicted to be on topic
      on_topic_pred = self.df[self.df.index.isin(self.hf_pred_docs)]
      on_topic_pred.to_csv(f'{self.directory}/test_docs_pred_on_topic.csv')

      #### Output the prediction metrics into a txt file #####
      output_filename = 'test_prediction_metrics.txt'
      file_path_metrics = os.path.join(self.directory, output_filename)

      #create output directory if it doesn't exist
      if not os.path.exists(self.directory):
        os.mkdir(self.directory)

      with open(file_path_metrics, 'w') as file:
        #accuracy score
        self.accuracy_score = accuracy_score(self.y_test, self.y_pred)
        file.write(f'Accuracy Score: {self.accuracy_score}\n\n')

        # Calculate Precision
        self.precision = precision_score(self.y_test, self.y_pred, average='macro')  # 'macro' average is one of the options
        file.write(f'Precision: {self.precision}\n\n')

        # Calculate Recall
        self.recall = recall_score(self.y_test, self.y_pred, average='macro')
        file.write(f'Recall: {self.recall}\n\n')

        # Calculate F1 score
        self.f1 = f1_score(self.y_test, self.y_pred, average='macro')
        file.write(f'F1 Score: {self.f1}\n\n')

        #classification report
        report = classification_report(self.y_test, self.y_pred)
        file.write(report)

        file.close()

      ### Confusion matrix ###
      cm = confusion_matrix(self.y_test, self.y_pred)
      # Create a plot
      fig, ax = plt.subplots()
      ConfusionMatrixDisplay(cm).plot(ax=ax)
      # Save the plot to a file
      plt.savefig(f'{self.directory}/confusion_matrix.png')
      # Close the plot to free memory
      plt.close(fig)

    def predict_unlabeled(self, unlabeled_docs):
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
      output_file = 'unlabeled_docs_prediction.csv'
      file_path = os.path.join(self.directory, output_file)
      df.to_csv(file_path)

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Naive Bayes Classifier for PubMed Documents')

    # Add arguments
    parser.add_argument('--num_classes', type=int, default = 2, help='Number of classes')
    parser.add_argument('--off_topic_class', type=int, help='Class that contains documents not in disease of interest')
    parser.add_argument('--input_path', type=str, help='Path to input CSV file')
    parser.add_argument('--out_path', type=str, default='./output', help='Path for output directory (default is ./output)')
    parser.add_argument('--unlabeled_docs_path', type=str, help='Path for unlabeled documents CSV file that contains a column with title abstract')
    parser.add_argument('--run_mode', type=str, choices=['train_test', 'predict_unlabeled'], required=True, help='Mode to run the script: "train_test" or "predict_unlabeled"')

    # Parse the arguments
    args = parser.parse_args()

    # Create an instance of the NBC class with the provided arguments
    nbc = NBC(args.num_classes, args.off_topic_class, args.input_path, args.out_path)

    if args.run_mode == 'train_test':
        nbc.prepare_feature_matrix()
        nbc.prior_c()
        nbc.train()
        nbc.predict_test_split()

    elif args.run_mode == 'predict_unlabeled':
        if not args.unlabeled_docs_path:
            raise ValueError("Unlabeled documents path is required for 'predict_unlabeled' mode")
        nbc.prepare_feature_matrix()
        nbc.prior_c()
        nbc.train()
        nbc.predict_test_split()
        nbc.predict_unlabeled(args.unlabeled_docs_path)

if __name__ == "__main__":
    main()