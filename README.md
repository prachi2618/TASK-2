
# Sentiment Analysis on Customer Reviews

This project performs sentiment analysis on a dataset of customer reviews using TF-IDF vectorization and a Logistic Regression classifier. The objective is to predict the sentiment of a review based on its content, with sentiments categorized as "Positive", "Negative", and "Irrelevant".

## Project Overview

Sentiment analysis is a Natural Language Processing (NLP) task that aims to classify the sentiment conveyed in a text (e.g., a review) into predefined categories. In this project, we use a dataset containing customer reviews and their corresponding sentiments. We preprocess the data, convert the text data into numerical features using TF-IDF, and train a Logistic Regression model to classify the sentiments.

## Files

- `twitter_training.csv`: The dataset containing customer reviews and sentiment labels.
- `sentiment_analysis.py`: The Python script that performs sentiment analysis.
- `README.md`: Documentation about the project.

## Requirements

This project requires the following Python libraries:

- `numpy`
- `pandas`
- `sklearn`
- `matplotlib`

You can install these dependencies using pip:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Data Preprocessing

1. **Loading the Data**: The dataset is loaded into a Pandas DataFrame using the `pd.read_csv()` function.
   
2. **Sampling the Data**: To make the dataset smaller for faster processing, a random sample of 10% of the data is selected using `df.sample()`.

3. **Selecting Required Columns**: Only the `Sentiment` and `Reviews` columns are kept, which contain the sentiment labels and the corresponding reviews.

4. **Handling Missing Data**: Any rows with missing values are dropped using `data.dropna()`.

## Model Training

1. **Text Vectorization**: The reviews (text data) are converted into numerical features using the TF-IDF vectorizer (`TfidfVectorizer` from `sklearn`). This process captures the importance of words in the context of the entire dataset.
   
2. **Splitting the Data**: The dataset is split into training and testing sets using `train_test_split()` from `sklearn`.

3. **Training the Logistic Regression Model**: A Logistic Regression model is trained on the training set to classify the sentiment of reviews.

4. **Evaluation**: After the model is trained, it is tested on the test set. The accuracy of the model is calculated using `accuracy_score()`, and the classification report is displayed using `classification_report()`.

## Results

The accuracy and classification report of the model are printed after evaluation. However, due to possible data imbalance or model limitations, the accuracy might be low. It is recommended to consider improving the model by using techniques like hyperparameter tuning, balancing the dataset, or experimenting with different models.

## Future Improvements

1. **Hyperparameter Tuning**: Use grid search or random search to tune the hyperparameters of the Logistic Regression model.
   
2. **Handling Data Imbalance**: Implement techniques like oversampling/undersampling or use different evaluation metrics such as F1-score to deal with class imbalance.

3. **Advanced Models**: Consider using more advanced machine learning models, such as Support Vector Machines (SVM) or deep learning models (e.g., CNNs or LSTMs), for better performance on text data.

4. **Text Preprocessing**: Experiment with additional preprocessing steps, such as removing stopwords, stemming, and lemmatization, to improve feature extraction.

## Conclusion

This project demonstrates how to perform sentiment analysis on customer reviews using TF-IDF vectorization and Logistic Regression. While the initial model might not perform well, various improvements can be made to enhance its accuracy and performance.

---

