# Hate Speech Classification Project

This project focuses on classifying social media comments into three categories: normal, offensive, and hatespeech. The classification is based on a dataset containing various comments and metadata, which is processed using natural language processing (NLP) techniques.

## Dataset

The dataset used in this project consists of two files:

1. `hateXplain.csv`: Original dataset with detailed information about comments.
2. `final_hateXplain.csv`: A cleaned version of the dataset, with a focus on relevant columns such as `comment`, `label`, and various demographic attributes.

### Columns in `final_hateXplain.csv`
- **comment**: The text of the comment.
- **label**: The classification label for the comment, which can be one of the following:
  - `normal`
  - `offensive`
  - `hatespeech`
- **Race**: The race of the person making the comment.
- **Religion**: The religion of the person making the comment.
- **Gender**: The gender of the person making the comment.
- **Sexual Orientation**: The sexual orientation of the person making the comment.
- **Miscellaneous**: Additional information about the person making the comment (e.g., `Unknown`, `Other`).

## Installation

To run this project, you'll need to have Python installed, as well as the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- nltk
- scikit-learn
- wordcloud

You can install the required dependencies using `pip`:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud
```

## Steps Involved

### Data Preprocessing

1. **Missing Value Handling**: We handle missing values in the `Miscellaneous` column by filling them with the label `'Unknown'`.
2. **Data Transformation**: Text data is tokenized using the `word_tokenize` function from NLTK. We then apply both `CountVectorizer` and `TfidfVectorizer` for feature extraction.

### Exploratory Data Analysis (EDA)

- Visualizing the distribution of labels (`normal`, `offensive`, `hatespeech`).
- Analyzing the demographic columns (`Race`, `Religion`, `Gender`, `Sexual Orientation`, `Miscellaneous`).

### Feature Creation

We create two types of features:
1. **CountVectorizer**: Converts a collection of text documents to a matrix of token counts.
2. **TfidfVectorizer**: Converts text data into a matrix of term-frequency times inverse document-frequency.

### Model Training

- We use an **MLPClassifier (Multilayer Perceptron)** model for classification. This model is trained on the features generated from the text data.

### Performance Evaluation

- After training the model, we evaluate its performance using:
  - **Classification Report**: Precision, recall, f1-score for each class.
  - **Confusion Matrix**: Visualized using `seaborn` to show true vs predicted labels.

### Model Usage

The model can be used to classify new input comments into the categories `normal`, `offensive`, or `hatespeech`. Here's an example of how to use the model:

```python
def print_prediction(input_text):
    result = play(input_text)
    print(f"\n{'='*30}\nInput Text: {input_text}\nPredicted Label: {result}\n{'='*30}")
```

Example predictions:

```bash
Input Text: you are beautiful
Predicted Label: normal

Input Text: you are stupid
Predicted Label: hatespeech

Input Text: i dont like you
Predicted Label: normal

Input Text: are you Muslim?
Predicted Label: offensive

Input Text: fuck you
Predicted Label: hatespeech
```

## Model Accuracy

The trained model achieves an accuracy of **93%** on the test data, and the classification report shows high precision, recall, and f1-scores across all classes.

## Conclusion

This project demonstrates a simple yet effective approach to classifying hate speech and offensive language in social media comments using machine learning techniques, specifically **MLPClassifier** with feature extraction through **CountVectorizer** and **TfidfVectorizer**.
