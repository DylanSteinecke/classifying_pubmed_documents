from datasets import Dataset, load_dataset, ClassLabel
import pandas as pd
import numpy as np
import time
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel, get_scheduler
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.optim import AdamW 
from torch.utils.data import DataLoader

from sklearn.metrics import precision_recall_fscore_support, brier_score_loss, classification_report
from sklearn.model_selection import train_test_split

from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import calibration_curve




def prepare_feature_matrix(feature_matrix_path, use_head):
    mat = pd.read_csv(feature_matrix_path)

    # Filter for documents with abstracts
    try:
        mat_filt = mat[~mat[topic_labels.str.contains(',')]]
    except:
        mat_filt = mat
        pass
    
    mat_filt = mat_filt.dropna(subset=['abstract'])

    # Filter for PMIDs with one label
    mat_filt['topic_labels'] = mat_filt['topic_labels'].astype(int)

    # Concatenate titles and abstracts
    mat_filt['abstract'] = mat_filt['title'] + ' ' + mat_filt['abstract']

    # Use either the first rows (use_head) or the whole matrix
    if use_head:
        other_label = max([int(label) for label in mat_filt['topic_labels'].tolist()])
        head_len = len(mat_filt) - mat_filt['topic_labels'].value_counts()[other_label] * 2
        if head_len < 0:
            head_len = len(mat_filt)
        my_matrix = mat_filt[['abstract', 'topic_labels']].head(head_len)
    else:
        # use all rows
        my_matrix = mat_filt[['abstract', 'topic_labels']]

    # Convert the filtered dataframe to a HuggingFace Dataset
    my_dataset = (Dataset.from_pandas(my_matrix)
                  .rename_column('topic_labels', 'labels')
                  .remove_columns('__index_level_0__'))

    # Split the dataset manually using scikit-learn
    X = my_dataset['abstract']
    y = my_dataset['labels']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Create train and test datasets
    train_dataset = Dataset.from_pandas(pd.DataFrame({'abstract': X_train, 'labels': y_train}))
    test_dataset = Dataset.from_pandas(pd.DataFrame({'abstract': X_test, 'labels': y_test}))

    # Combine train and test datasets into a dictionary
    combined_dataset = {'train': train_dataset, 'test': test_dataset}

    return combined_dataset


def convert_hf_ft_matrix_to_two_classes(input_file):
    df_6 = pd.read_csv(f'output/{input_file}.csv')
    df_6['topic_labels'] = df_6['topic_labels'].replace(['0','1','2','3','4'], '0')
    df_6['topic_labels'] = df_6['topic_labels'].replace('5', '1')
    df_2 = df_6
    df_2.to_csv(f'input/{input_file}_2_classes.csv')

    
    

def compute_metrics(logits, labels, num_labels):
    labels = torch.tensor(labels)
    softmax = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    predictions = np.argmax(softmax, axis=-1).clone().detach()

    # Precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
                                  labels, predictions, average='macro')
    conf_matrix = classification_report(labels, 
                                    predictions, 
                                    target_names=[str(label) for label in range(num_labels)])
    
    # Accuracy
    correct = {i: 0 for i in range(num_labels)}
    total = {i: 0 for i in range(num_labels)}
    with torch.no_grad():
        for i, (lbls, predicts) in enumerate(zip(labels, predictions)):
            for label_j in range(num_labels):
                correct[label_j] += ((predicts==label_j) & (lbls==label_j)).sum().item()
                total[label_j] += (lbls==label_j).sum().item()
        accuracy = {i: correct[i] / total[i] for i in range(num_labels)}

    # Final evaluation metrics
    eval_metrics = {'precision': precision,   # Precision (average over classes)
                    'recall': recall,         # Recall (average over classes)
                    'f1': f1,                 # F1-score (average over classes)
                    'confusion_matrix': conf_matrix, # Class-specific precision, recall, F1
                   }
    for label_num in range(num_labels):       # Class-specific accuracy
        eval_metrics[f'acc_{label_num}'] = accuracy[label_num]
        
    return eval_metrics
        

def flatten_list(the_list):
    flat_list = [each_item for each_list in the_list for each_item in each_list ]
    return flat_list
    
    
   ## fix for multi class
def plot_probability_calibration_curve(logits, labels, model_name):
    probabilities = torch.nn.functional.softmax(torch.tensor(logits).cpu(), dim=-1)
    positive_probabilities = probabilities[:,1]

    b_score = brier_score_loss(labels, positive_probabilities)
    print("Brier Score :",b_score)

    # True and Predicted Probabilities
    true_pos, pred_pos = calibration_curve(labels, 
                                           positive_probabilities, 
                                           n_bins=10)

    #Plot the Probabilities Calibrated curve
    plt.plot(pred_pos, true_pos, marker='o', linewidth=1, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.title('Probability Calibration Curve')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.legend(loc='best')
    plt.show()
    plt.savefig(f'output/prob_calibration_curve_{model_name}.png')
    
    # Plot a histogram of the predicted probability of a positive class
    plt.hist(positive_probabilities);
    plt.title('Positive Probabilities')
    plt.xlabel('Predicted Probability')
    plt.savefig(f'output/pred_prob_histogram_{model_name}.png')


class DocumentClassifier:
    
    def __init__(self, dataset):
        self.dataset = dataset
        
    def classify_documents(self, model_name, epochs,
                           num_labels, batch_size=16, 
                           model_name_suffix='', lr=3e-5, 
                           logfile='output/log.txt',
                           save_model=False):
        self.model_name = model_name
        self.epochs = epochs
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.model_name_suffix = model_name_suffix
        self.lr = lr
        self.logfile = logfile
        if '\\' and '_' in model_name: 
            model_name = model_name.split('/')[1].split('_')[0]
        model_file = f'{model_name}_{epochs}_epochs_{num_labels}_classes_{model_name_suffix}'
        
        ''' 
        Base model
        '''
        model_name_to_checkpoint = {
          'bert':'bert-base-uncased',
          'biobert': 'dmis-lab/biobert-v1.1',
          'pubmedbert': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
          'biolink': 'michiyasunaga/BioLinkBERT-base',
          'specter': 'allenai/specter',
          'specter2': 'allenai/specter2',
        }
        if '/' in model_name:
            checkpoint = model_name
        else:
            checkpoint = model_name_to_checkpoint[model_name]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
        device = torch.device('cuda')
        model.to(device)

        ''' 
        Dataset
        '''
        dataset = self.dataset
        # Tokenize the train and test datasets separately
        train_tokenized = dataset["train"].map(lambda x: tokenizer(
                              x["abstract"],
                              truncation=True,
                              max_length=512, ))
        test_tokenized = dataset["test"].map(lambda x: tokenizer(
                              x["abstract"],
                              truncation=True,
                              max_length=512, ))

        # Merge the tokenized train and test datasets into a single dataset
        train_tokenized = train_tokenized.remove_columns('abstract').with_format('torch')
        test_tokenized = test_tokenized.remove_columns('abstract').with_format('torch')
        tokenized_datasets = {"train": train_tokenized, "test": test_tokenized}
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Turn dataset into a PyTorch dataloader
        train_dataloader = DataLoader(
                                tokenized_datasets['train'], 
                                shuffle=True, # why is this shuffled 
                                batch_size=batch_size, 
                                collate_fn=data_collator,)
        eval_dataloader = DataLoader(
                                tokenized_datasets['test'], 
                                batch_size=batch_size, 
                                collate_fn=data_collator,)

        ''' 
        Hyperparameters 
        '''
        optimizer = AdamW(model.parameters(), lr=lr)
        num_training_steps = epochs*len(train_dataloader)
        progress_bar = tqdm(range(num_training_steps))
        lr_scheduler = get_scheduler(
                              'linear',
                              optimizer=optimizer,
                              num_warmup_steps=0,
                              num_training_steps=num_training_steps,)

        # Loss function
        classes = [num for num in range(num_labels)]
        y = tokenized_datasets['train']['labels'].tolist()
        weights = compute_class_weight('balanced', classes=classes, y=y)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)


        '''
        Model training
        '''
        with open(logfile,'w') as fout_log:

            ### Training Loop ###
            best_val_loss = 999999999
            all_train_evals, all_val_evals = {}, {}
            for epoch in range(epochs):
                total_train_loss, total_val_loss = 0, 0
                train_logits, train_labels = [], []
                val_logits, val_labels = [], []

                ''' 
                Training
                '''
                model.train()
                for batch_num, batch in enumerate(train_dataloader):
                    batch = {k:v.to(device) for k,v in batch.items()}
                    outputs = model(**batch)
                    logits = outputs.logits
                    labels = batch['labels']
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    total_train_loss += loss.item()
                    train_logits.extend(logits.tolist())
                    train_labels.extend([l.item() for l in labels])


                ''' 
                Validation
                '''
                model.eval()
                with torch.no_grad():
                    for val_batch_num, val_batch in enumerate(eval_dataloader):
                        val_batch = {k:v.to(device) for k,v in val_batch.items()}
                        outputs = model(**val_batch)

                        # Validation Loss (Batch)
                        logits = outputs.logits
                        labels = val_batch['labels']
                        val_loss = criterion(logits, labels)
                        total_val_loss += val_loss

                        # Validation Metrics (Batch)
                        logits = outputs.get("logits")
                        val_logits.extend(logits.tolist())
                        val_labels.extend([l.item() for l in labels])


                '''
                Evaluation metrics
                '''
                # Training evaluation metrics
                train_metrics = compute_metrics(train_logits, train_labels, num_labels)
                print('Training Set Metrics')
                for metric_name, metric_num in train_metrics.items():
                    print(metric_name, metric_num)
                    fout_log.write(f'{metric_name} {metric_num}')
                    all_train_evals.setdefault(metric_name, []).append(metric_num)

                avg_train_loss = total_train_loss/len(train_dataloader)
                all_train_evals.setdefault('train_loss', []).append(avg_train_loss)

                # Validation evaluation metrics
                val_metrics = compute_metrics(val_logits, val_labels, num_labels)
                print('Validation Set Metrics')
                for metric_name, metric_num in val_metrics.items():
                    print(metric_name, metric_num)
                    fout_log.write(f'{metric_name} {metric_num}')
                    all_val_evals.setdefault(metric_name, []).append(metric_num)

                avg_val_loss = total_val_loss/len(eval_dataloader)
                all_val_evals.setdefault('val_loss', []).append(avg_val_loss)
                print(f'Epoch {epoch} | Training Loss: {avg_train_loss} | Validation Loss: {avg_val_loss}')
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model_path = f'output/epoch_{epoch}_{model_name}_{epochs}_epochs_{num_labels}_classes_{model_name_suffix}'
                    torch.save(model.state_dict(), model_path)

        # Save logits (predictions, almost) and labels (true answers)
        with torch.no_grad():
            train_embeddings, val_embeddings = [], []
            train_labels, val_labels = [], []
            for batch in train_dataloader:
                batch = {k:v.to(device) for k,v in batch.items()}
                outputs = model(**batch)
                train_embeddings.append(outputs[0])
                labels = batch['labels']
                train_labels.append(labels)

            for batch in eval_dataloader:
                batch = {k:v.to(device) for k,v in batch.items()}
                outputs = model(**batch)
                val_embeddings.append(outputs[0])
                labels = batch['labels']
                val_labels.append(labels)
            
        if num_labels == 2:
            flat_val_labels = flatten_list([label.to('cpu').tolist() for label in val_labels])
            plot_probability_calibration_curve(val_logits, flat_val_labels, model_name)
        

        # Save model
        if save_model:
            model.save_pretrained(f'output/{model_file}')
        
        
        self.all_train_evals = all_train_evals
        self.all_val_evals = all_val_evals
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.train_embeddings = train_embeddings
        self.val_embeddings = val_embeddings
     