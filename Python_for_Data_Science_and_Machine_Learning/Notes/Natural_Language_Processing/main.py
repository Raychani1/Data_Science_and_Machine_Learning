import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
