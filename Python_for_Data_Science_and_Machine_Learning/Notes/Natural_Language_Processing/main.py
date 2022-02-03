import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from typing import List
from sklearn.feature_extraction.text import CountVectorizer

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
