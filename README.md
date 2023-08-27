# Anxiety_detection

The provided Python code performs a comprehensive text analysis and classification task using various libraries. It begins by loading text data from an Excel file and preprocesses it by removing missing values. The distribution of labels is visualized through pie and bar charts. The text content is analyzed by calculating word and character counts, and their distributions are visualized using KDE plots.
Here's a breakdown of what the code does:

1. **Importing Libraries**: The code begins by importing necessary libraries for data manipulation, visualization, natural language processing, and machine learning.

2. **Loading Data**: The code loads data from an Excel file using the `pd.read_excel` function and displays the first few rows using the `head()` function.

3. **Data Preprocessing and Analysis**:
   - The code checks the shape of the data using `print(data.shape)`.
   - It checks for missing values in the dataset using `data.isnull().sum()` and drops rows with any missing values using `data=data.dropna(how='any')`.
   - The distribution of labels is analyzed using value counts and visualized using a pie chart and a bar plot.

4. **Text Analysis**:
   - The code calculates the total number of words and characters in each text and adds these as new columns to the dataframe.
   - It visualizes the distribution of total words and total characters using KDE plots.

5. **Text Preprocessing**:
   - Text is converted to lowercase using the `convert_lowercase` function.
   - URLs are removed from the text using the `remove_url` function.
   - Punctuation is removed using the `remove_punc` function.
   - Stopwords are removed from the text using the `remove_stopwords` function.
   - The total number of words after these transformations is calculated and added to the dataframe.

6. **Word Clouds**:
   - Word clouds are generated separately for both classes (0 and 1) to visualize the most common words in each class.

7. **Most Common Words Analysis**:
   - The code analyzes the most common words used in both classes by creating bar plots of word frequencies.

8. **Feature Extraction and Model Training**:
   - The text data is split into features (X) and labels (y).
   - The dataset is split into training and testing sets using `train_test_split`.
   - Text data is transformed into TF-IDF features using `TfidfVectorizer`.
   - The `train_model` function trains and evaluates both a Multinomial Naive Bayes classifier and a Random Forest classifier.
