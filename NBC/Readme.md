Overview
NBC.py is a Python script implementing a Naive Bayes Classifier (NBC) tailored for categorizing PubMed documents. It is designed to classify documents into specified categories based on their content, particularly useful in biomedical literature analysis.

Prerequisites
Before running the script, ensure you have Python installed on your system. The script has been tested with Python 3.8+. Additionally, the required Python packages can be installed via the requirements.txt file.

Installation
First, clone the repository or download the NBC.py script and the requirements.txt file. Install the necessary dependencies by running:

Copy code
pip install -r requirements.txt
Input Data Format
The script expects a CSV file as input with specific columns:

title: Title of the PubMed document.
abstract: Abstract of the document.
topic_labels: Labels indicating the document's category.
Run Modes
NBC.py supports two primary run modes:

train_test: This mode trains the Naive Bayes Classifier on a provided dataset and evaluates its performance on a test split.
predict_unlabeled: In this mode, the script predicts the categories of unlabeled documents.
Command-Line Arguments
The script accepts several command-line arguments to control its behavior:

--num_classes (default=2): The number of classes for classification.
--off_topic_class: Specifies the class representing documents not in the disease of interest.
--input_path: The path to the input CSV file containing labeled data.
--out_path (default='./output'): The directory where output files will be saved.
--unlabeled_docs_path: Path for the CSV file containing unlabeled documents (required for predict_unlabeled mode).
--run_mode (required): The mode to run the script in, either train_test or predict_unlabeled.
How to Run
To execute the script, navigate to the directory containing NBC.py and run it via the terminal. Here are example commands for each mode:

For train_test mode:

css
Copy code
python NBC.py --run_mode train_test --input_path path/to/input.csv --off_topic_class 0
For predict_unlabeled mode:

css
Copy code
python NBC.py --run_mode predict_unlabeled --input_path path/to/input.csv --off_topic_class 0 --unlabeled_docs_path path/to/unlabeled_docs.csv
Output
The script generates various outputs based on the chosen mode, including:

Word counts and log likelihoods for each class.
Classification reports and confusion matrices for the train_test mode.
Predicted categories for unlabeled documents in predict_unlabeled mode.
All outputs are saved in the directory specified by --out_path.

Additional Information
Ensure that your input CSV files are correctly formatted as per the Input Data Format section. For best results, clean and pre-process your data accordingly.