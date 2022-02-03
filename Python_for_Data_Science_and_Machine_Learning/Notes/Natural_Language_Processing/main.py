import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from typing import List
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':

    # Setup

    # We need to download the 'stopwords' package
    # d -> stopwords
    # nltk.download_shell()

    # Simple SMS Spam Detection Filter

    # Read in Data to the corpus (Collection of Text )with simple list
    # comprehension
    messages = [line.rstrip() for line in open('data/SMSSpamCollection')]

    # Display the number of messages we have
    print(len(messages), end='\n\n')

    # Display the first 10 messages through enumeration
    for mess_no, message in enumerate(messages[:10]):
        print(mess_no, message, end='\n\n')

    # We can see that our data is TSV (Tab Separated Values)
    print(messages[0])

    # Separate the label from message and save them to DataFrame
    messages = pd.read_csv(
        'data/SMSSpamCollection',
        sep='\t',
        names=['label', 'message']
    )

    # Display basic information about the data set
    print(messages.head(), end='\n\n')

    print(messages.describe(), end='\n\n')

    print(messages.groupby('label').describe(), end='\n\n')

    messages['length'] = messages['message'].apply(len)

    print(messages.head(), end='\n\n')

    # Exploratory Data Analysis

    # We can see there are some long messages
    messages['length'].plot.hist(bins=150)
    plt.show()

    # Let's have a look
    print(messages['length'].describe(), end='\n\n')

    # Display the longest message
    print(messages[messages['length'] == 910]['message'].iloc[0])

    # We can see that ham messages tend to have shorter form, and spam messages
    # are a bit longer
    messages.hist(column='length', by='label', bins=60, figsize=(16, 9))
    plt.show()

    # Example to remove punctuation from text
    mess = 'Sample message! Notice : it has punctuation.'

    nopunc = ''.join([c for c in mess if c not in string.punctuation])

    print(nopunc, end='\n\n')

    # English Stopwords ( common not relevant words )
    print(stopwords.words('english'), end='\n\n')

    # Remove stopwords from text without punctuation
    clean_mess = [
        word for word in nopunc.split() if
        word.lower() not in stopwords.words('english')
    ]

    print(clean_mess, end='\n\n')


    def text_process(message_text: str) -> List[str]:
        """Removes punctuation and stopwords from message.

        Args:
            message_text (str): Message text

        Returns:
            List[str]: Clean text words
        """

        # Remove Punctuation
        no_punc = ''.join(
            [char for char in message_text if char not in string.punctuation]
        )

        # Remove Stopwords
        return [
            word for word in no_punc.split() if
            word.lower() not in stopwords.words('english')
        ]


    print(messages['message'].head(5).apply(text_process), end='\n\n')

    # This is just basic preprocessing

    # Stemming allows replacing similar words that point to the same thing
    # with a single word, for this we need a reference dictionary.
    # However, stemming won't work really well on shorthand words
    # ( U, Nah, etc.)

    # Next we will need to create a Count Vector from all of our messages.
    # Which is basically an Every_word x Number_of_Messages Matrix, and each
    # line represents, that how many times is a given word in a given message.

    bow_transformer = CountVectorizer(analyzer=text_process).fit(
        messages['message']
    )

    # We can check how many words do we have in our vocabulary
    print(len(bow_transformer.vocabulary_))

    # We can now transform a message
    mess4 = messages['message'][3]

    # Display the original text message
    print(mess4, end='\n\n')

    # Transform it
    bow4 = bow_transformer.transform([mess4])

    # We get a Count Vector representation
    print(bow4, end='\n\n')

    # We can also check the shape of our vector
    print(bow4.shape, end='\n\n')

    # We saw in our vector that some words appear multiple times in the text,
    # we can check which are those words ( The index is from our Count Vector ).
    print(bow_transformer.get_feature_names_out()[4068], end='\n\n')
    print(bow_transformer.get_feature_names_out()[9554], end='\n\n')

    # We transform all the messages
    messages_bow = bow_transformer.transform(messages['message'])

    print(f'Shape of the Sparse Matrix: {messages_bow.shape}', end='\n\n')

    # We can check all the non-zero occurrences
    print(messages_bow.nnz, end='\n\n')

    # We can compare the number of non-zero elements to all the elements
    sparsity = (
            100.0 * messages_bow.nnz /
            (messages_bow.shape[0] * messages_bow.shape[1])
    )

    print(f'Sparsity: {sparsity}', end='\n\n')

    # For weights, we are going to use the Term Frequency - Inverse Document
    # Frequency (TF-IDF)
    tfidf_transformer = TfidfTransformer().fit(messages_bow)

    # Transforming only single message
    tfidf4 = tfidf_transformer.transform(bow4)

    print(tfidf4, end='\n\n')

    # We can check the document frequency of any word
    print(
        tfidf_transformer.idf_[bow_transformer.vocabulary_['university']],
        end='\n\n'
    )

    # Transform every message
    messages_tfidf = tfidf_transformer.transform(messages_bow)

    # For our spam detector we are going to use the Multinomial Naive Bayes
    # Algorithm
    spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

    # Prediction for single message
    print(
        f"Real Label: {messages['label'][3]}\n"
        f"Predicted Label: {spam_detect_model.predict(tfidf4)[0]}"
    )

    # Prediction for every message
    all_pred = spam_detect_model.predict(messages_tfidf)

    # We need to note that we are predicting on the same data set as the
    # training data set. So the accuracy of our model is not precise.

    # First we split the data to train and test set
    msg_train, msg_test, label_train, label_test = train_test_split(
        messages['message'], messages['label'], test_size=0.3
    )

    # Scikit-Learn supports Pipeline Features

    # pipeline = Pipeline([
    #     ('bow', CountVectorizer(analyzer=text_process)),
    #     ('tfidf', TfidfTransformer()),
    #     ('classifier', MultinomialNB())
    # ])

    # We can also use other types of Classifiers
    pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=text_process)),
        ('tfidf', TfidfTransformer()),
        ('classifier', RandomForestClassifier())
    ])

    pipeline.fit(msg_train, label_train)

    predictions = pipeline.predict(msg_test)

    print(classification_report(label_test, predictions), end='\n\n')
