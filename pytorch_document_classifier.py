from datasets import Dataset, load_dataset, ClassLabel
import pandas as pd
import numpy as np
import time
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Transformers
from transformers import AutoTokenizer, AutoModel, get_scheduler
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding

# PyTorch
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.optim import AdamW 
from torch.utils.data import DataLoader

# Sklearn
from sklearn.metrics import precision_recall_fscore_support, brier_score_loss, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import calibration_curve




def prepare_feature_matrix(feature_matrix_path, test_size=0.5, upsample_train=True):
    df = pd.read_csv(feature_matrix_path)

    # Remove documents without an abstract
    df = df.dropna(subset=['abstract'])

    # Concatenate titles and abstracts
    if 'title' in df.columns:
        df.loc[:, 'abstract'] = df['title']+' '+df['abstract']
        df = df.drop('title', axis=1) 

    # Remove documents with more than 1 label of interest
    df = df[~df['labels'].astype(str).str.contains(',')]    
    df.loc[df['labels'].apply(lambda x: isinstance(x, int))] 
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Create train and test datasets
    X = df['abstract']
    y = df['labels']
    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=test_size, stratify=y))

    # Train dataset
    train_df = pd.DataFrame({'abstract': X_train, 'labels': y_train}).reset_index(drop=True)
    if upsample_train:
        train_df = upsample_dataset(train_df, 'labels') # note: tokenize first for speed?
    train_dataset = Dataset.from_pandas(train_df)

    # Test dataset
    test_df = pd.DataFrame({'abstract': X_test, 'labels': y_test}).reset_index(drop=True)
    test_dataset = Dataset.from_pandas(test_df)

    # Combine train and test datasets into a dictionary
    combined_dataset = {'train': train_dataset, 
                        'test': test_dataset}

    return combined_dataset


def upsample_dataset(df, label_col='labels'):
    '''For upsampling the training dataset to make classes balanced'''
    num_classes = len(set(df[label_col]))
    positive_examples = df[df[label_col] < num_classes-1]
    negative_examples = df[df[label_col] == num_classes-1]
    
    # Upsample positive examples if there are less of them
    if len(positive_examples) < len(negative_examples) :
        positive_upsampled = resample(positive_examples,
                                      replace=True,
                                      n_samples=len(negative_examples),
                                      random_state=316)
        upsampled_df = pd.concat([negative_examples, 
                                  positive_upsampled])
    # Upsample negative examples if there are less of them
    elif len(negative_examples) < len(positive_examples):
        negative_upsampled = resample(negative_examples,
                                      replace=True,
                                      n_samples=len(positive_examples),
                                      random_state=316)
        upsampled_df = pd.concat([positive_examples, 
                                  negative_upsampled])

    upsampled_df = upsampled_df.sample(frac=1, random_state=316).reset_index(drop=True)
    
    return upsampled_df


# Currently this is too hardcoded. Also, it is intended for multi-class classification. 
def convert_hf_ft_matrix_to_two_classes(input_file):
    df_6 = pd.read_csv(f'output/{input_file}.csv')
    df_6['labels'] = df_6['labels'].replace(['0','1','2','3','4'], '0')
    df_6['labels'] = df_6['labels'].replace('5', '1')
    df_2 = df_6
    df_2.to_csv(f'input/{input_file}_2_classes.csv', index=False)

    
    

def compute_metrics(logits, labels, num_labels):
    labels = torch.tensor(labels)
    softmax = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    predictions = np.argmax(softmax, axis=-1).clone().detach()

    # Precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
                                  labels, predictions, average='macro') # Dylan, question?
    target_names = [str(label) for label in range(num_labels)]
    print('num_labels', num_labels)
    print('labels', labels)
    print('target_names', target_names)
    conf_matrix = classification_report(labels, 
                                    predictions, 
                                    target_names=target_names) # if number of classes does not match target_names size, it may be because there are not any examples predicted correctly in a certain class
    
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
    
    



class DocumentClassifier:
    
    def __init__(self, dataset, topic, stage_num):
        self.dataset = dataset
        self.topic = topic
        self.out_dir = f'output/{topic}'
        self.stage_num = stage_num
        
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
        plt.savefig(f'{self.out_dir}/prob_calibration_curve_{model_name}.png')

        # Plot a histogram of the predicted probability of a positive class
        plt.hist(positive_probabilities);
        plt.title('Positive Probabilities')
        plt.xlabel('Predicted Probability')
        plt.savefig(f'{self.out_dir}/pred_prob_histogram_{model_name}.png')
        
        
    def classify_documents(self, model_name, epochs,
                           num_labels, batch_size=16, 
                           model_name_suffix='', lr=3e-5, 
                           save_model=False):
        logfile=f'{self.out_dir}/{model_name}_{model_name_suffix}_log.txt'
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
        train_tokenized = dataset['train'].map(lambda x: tokenizer(
                              x['abstract'],
                              truncation=True,
                              max_length=512,))
        test_tokenized = dataset['test'].map(lambda x: tokenizer(
                              x['abstract'],
                              truncation=True,
                              max_length=512,))

        # Merge the tokenized train and test datasets into a single dataset
        train_tokenized = train_tokenized.remove_columns('abstract').with_format('torch')
        test_tokenized = test_tokenized.remove_columns('abstract').with_format('torch')
        tokenized_datasets = {'train': train_tokenized, 
                              'test': test_tokenized}
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
        Model training & testing
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
                Testing/Validation
                '''
                model.eval()
                val_examples, val_predicted_labels = [], []
                with torch.no_grad():
                    for val_batch_num, val_batch in enumerate(eval_dataloader):
                        val_batch = {k:v.to(device) for k,v in val_batch.items()}
                        outputs = model(**val_batch)
                        logits = outputs.logits

                        # Validation Loss (Batch)
                        labels = val_batch['labels']
                        val_loss = criterion(logits, labels)
                        total_val_loss += val_loss

                        # Validation Metrics (Batch)
                        input_ids = val_batch['input_ids']
                        input_ids = input_ids.cpu()
                        val_batch_examples = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                        val_examples.extend(val_batch_examples)              # Text
                        val_labels.extend([l.item() for l in labels.cpu()])  # True labels
                        val_logits.extend(logits.tolist())
                        val_batch_preds = torch.argmax(logits, dim=1).cpu()
                        val_batch_preds = [pred.item() for pred in val_batch_preds]
                        val_predicted_labels.extend(val_batch_preds)   # Predicted labels

                  
                # Export examples (validation set, positive predicted subset of val. set)
                val_set = pd.DataFrame({'abstract':val_examples, 
                                        'true_label':val_labels, 
                                        'predicted_label':val_predicted_labels,})
                val_set.to_csv(f'{self.out_dir}/{self.topic}_val_set_epoch_{epoch}.csv', 
                               index=False)
                        
                pos_classes = list(range(0, num_labels-1))
                pos_pred_val_set = val_set[val_set['predicted_label'].isin(pos_classes)]
                pos_pred_val_set = pos_pred_val_set.drop('predicted_label', axis=1)
                outpath = f'{self.out_dir}/stage_{self.stage_num}_test_docs_pred_ontopic_{self.topic}.csv'
                pos_pred_val_set.to_csv(outpath, index=False)

                '''
                Evaluation metrics
                '''
                # Training evaluation metrics
                train_metrics = compute_metrics(train_logits, train_labels, num_labels)
                print('\n\nTraining Set Metrics')
                for metric_name, metric_num in train_metrics.items():
                    print(metric_name, metric_num)
                    fout_log.write(f'{metric_name} {metric_num}')
                    all_train_evals.setdefault(metric_name, []).append(metric_num)

                avg_train_loss = total_train_loss/len(train_dataloader)
                all_train_evals.setdefault('train_loss', []).append(avg_train_loss)

                # Validation evaluation metrics
                val_metrics = compute_metrics(val_logits, val_labels, num_labels)
                print('\n\nValidation Set Metrics')
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
            self.plot_probability_calibration_curve(val_logits, flat_val_labels, model_name)
        

        # Save model
        if save_model:
            model.save_pretrained(f'{self.out_dir}/{model_file}')
        
        
        self.all_train_evals = all_train_evals
        self.all_val_evals = all_val_evals
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.train_embeddings = train_embeddings
        self.val_embeddings = val_embeddings
     