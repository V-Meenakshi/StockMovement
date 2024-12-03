# *StockMovementAnalysis*

## *Project Description*  
StockMovementAnalysis aims to predict stock price movements by analyzing user-generated content from social media platforms like Twitter. By leveraging machine learning and sentiment analysis, this project forecasts stock trends, helping users gain insights into stock market fluctuations based on real-time discussions and sentiments.

---

## *Objective*  
To develop a machine learning model that:  
- Scrapes social media platforms such as Twitter, Reddit, or Telegram to gather user discussions about stocks.  
- Analyzes sentiments (positive, neutral, negative) in the data to predict stock price movements.  
- Provides actionable insights, forecasting whether a stock’s price is likely to increase, decrease, or remain stable.  

---

## *Features*  

### *1. Social Media Sentiment Analysis*  
- *Data Collection*:  
  - Fetches tweets using the Twitter API based on stock-related keywords and hashtags.  
  - Processes the text data to filter out noise (e.g., hashtags, emojis, and unnecessary punctuation).  

- *Sentiment Analysis*:  
  - Sentiments are categorized as *Positive, **Neutral, or **Negative* using *NLTK* (Natural Language Toolkit).  
  - Preprocessing includes tokenization, stopword removal, and lemmatization for cleaner text analysis.  
  - Generates insightful visualizations, such as WordClouds, to highlight frequent terms for each company.

---

### *2. Prediction*  
- *Sentiment-Based Rule System*:  
  - Predicts stock movements (e.g., "Increased approx 0.5%," "Decreased approx 0.5%," or "No significant change") based on the sentiment score derived from tweets.  

- *Machine Learning Models*:  
  - *Decision Trees (DT)*: Used initially for classification tasks.  
  - *Random Forest (RF)*: Chosen for its ability to enhance prediction accuracy by combining multiple decision trees.  

---

### *3. Query-Based Results*  
- Provides an interactive experience:  
  - Users can input a stock name and company name to get real-time predictions on stock movements.  
  - Displays sentiment and stock movement trends specific to the queried stock in a tabular format.  

---

### *4. Data Management*  
- *Rate Limit Handling*:  
  - Manages Twitter API restrictions by storing fetched data for reuse in training and analysis.  

- *Real-Time Integration*:  
  - Combines historical data with real-time tweets during prediction to ensure the results are relevant and up-to-date.  

---

## *Technologies Used*  

### *Programming Language*  
- *Python*  

### *Libraries*  
#### *Data Processing*  
- pandas  
- numpy  

#### *Visualization*  
- matplotlib  
- seaborn  
- wordcloud  
- plotly.express  
- plotly.graph_objects  

#### *Natural Language Processing*  
- nltk  
- spacy  

#### *Machine Learning*  
- scikit-learn  

#### *Web Scraping*  
- tweepy  

---

## *Algorithms Used*  
1. *Sentiment Analysis*:  
   - Sentiment scores were derived from tweets using Natural Language Processing (NLP) techniques.  

2. *Stock Movement Prediction*:  
   - *Decision Trees (DT)*: Used initially for classification tasks.  
   - *Random Forest (RF)*: Chosen for its ability to enhance prediction accuracy by combining multiple decision trees.  

---

## *Evaluation Metrics*  
The following metrics were used to assess model performance:  

1. *Accuracy*: Measures how often the model correctly predicts stock movement.  
2. *Confusion Matrix*: Provides a detailed view of prediction results by comparing actual and predicted labels.  
3. *Classification Report*: Includes metrics like precision, recall, and F1-score for each class.  

### *Evaluation Code Snippet*  
```python
models = [dt, rf]
accuracies = []

for model in models:
    print('Results for the model:', model._class.name_)
    model.fit(X_balanced, y_balanced)
    Y_train_bal = model.predict(X_balanced)
    y_pred = model.predict(X_test)

    training_accuracy = accuracy_score(Y_train_bal, y_balanced)
    print('Training Accuracy:', training_accuracy)

    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', cm)

    report = classification_report(y_test, y_pred)
    print('Classification Report:\n', report)

    print('\n')
    accuracies.append(accuracy)

print('List of Accuracies:', accuracies)

## **Challenges Faced**  

### **Rate Limiting**  
- Twitter API imposes restrictions on the number of tweets that can be fetched in a given timeframe.  
- To address this, we stored scraped data locally for training and used real-time tweets during prediction.  

### **Data Cleaning**  
- Processing noisy social media data with spelling errors, emojis, and hashtags posed challenges for effective analysis.  

### **Balancing Dataset**  
- Ensuring that positive, neutral, and negative sentiments were adequately represented for meaningful predictions.  

---

## **Setup Instructions**  

### **Prerequisites**  
1. Install Python (3.7 or higher).  
2. Install the required libraries using the `requirements.txt` file:  
   bash  
   pip install -r requirements.txt  
     
3. Obtain Twitter API credentials from [Twitter Developer Portal](https://developer.twitter.com/).  

### **Setup**  
1. Clone the repository:  
   bash  
   git clone https://github.com/yourusername/StockMovementAnalysis.git  
   cd StockMovementAnalysis  
     
2. Add the Twitter API credentials to the script.  
3. Ensure the dataset (`stock_tweets.csv`) is available in the project directory.  

---

## **Usage Instructions**  

### **Run the Prediction Script**  
Execute the Python script to predict stock movements based on sentiment analysis:  
bash  
python predict_stock_movement.py  
  

### **Query Stock Movements**  
Input stock and company names to get specific predictions:  
bash  
Enter the stock name (e.g., TSLA): TSLA  
Enter the company name (e.g., Tesla): Tesla  
  

### **Outputs**  
1. **CSV File**:  
   - The predictions will be saved in `stock_movement_predictions.csv`.  

2. **Console Results**:  
   - Displays filtered predictions for queried stocks and companies.  

---

## **Examples**  

### **Predicted Stock Movement**  

| Date       | Stock Name | Company Name | Sentiment | Sentiment_Score | Stock_Movement       |  
|------------|------------|--------------|-----------|------------------|----------------------|  
| 2024-12-01 | TSLA       | Tesla        | Positive  | 1                | Increased approx 0.5% |  
| 2024-12-01 | AAPL       | Apple        | Neutral   | 0                | No significant change |  
| 2024-12-01 | AMZN       | Amazon       | Negative  | -1               | Decreased approx 0.5% |  

---

## **Future Improvements**  
1. Incorporate data from additional platforms like Reddit and Telegram.  
2. Improve sentiment analysis accuracy by fine-tuning pre-trained models.  
3. Introduce a machine learning-based stock movement predictor.  

---

## **Contributing**  
Feel free to contribute by submitting pull requests or suggesting enhancements via issues.  
