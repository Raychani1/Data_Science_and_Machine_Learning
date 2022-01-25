import pandas as pd

if __name__ == '__main__':
    # Pandas setting(s)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    # Read Input Data
    ecom = pd.read_csv('data/Ecommerce_Purchases')

    # Display the head of the DataFrame
    print(ecom.head(), end='\n\n')

    # Display some information about our DataFrame
    print(ecom.info(), end='\n\n')

    # Display the average Purchase Price
    print(ecom['Purchase Price'].mean(), end='\n\n')

    # Display the highest Purchase Price
    print(ecom['Purchase Price'].max(), end='\n\n')

    # Display the lowest Purchase Price
    print(ecom['Purchase Price'].min(), end='\n\n')

    # Display the number of English users
    print(len(ecom[ecom['Language'] == 'en']), end='\n\n')

    # Display the number of Lawyers
    print(len(ecom[ecom['Job'] == 'Lawyer']), end='\n\n')

    # Display the number of purchases based on time
    print(ecom['AM or PM'].value_counts(), end='\n\n')

    # Display the Top 5 Most Common Job Titles
    print(ecom['Job'].value_counts().head(5), end='\n\n')

    # Display price from a specific lot
    print(ecom[ecom['Lot'] == '90 WT']['Purchase Price'], end='\n\n')

    # Display email of a specific customer
    print(ecom[ecom['Credit Card'] == 4926535242672853]['Email'], end='\n\n')

    # Display the number of customers with an American Express Credit Card
    # and a purchase above $95
    print(
        len(
            ecom[
                (ecom['CC Provider'] == 'American Express') &
                (ecom['Purchase Price'] > 95)
                ]
        ), end='\n\n'
    )

    # Display the number of users that have their credit card expiration in
    # 2025


    def expires_in_year(expiration_date: str, year: int) -> bool:
        """Returns boolean value if card expires in given year.

        Args:
            expiration_date (str): Expiration on Credit Card
            year (int): Year of expiration

        Returns:
            bool: Card expires in given year
        """

        return \
            True if list(map(int, expiration_date.split('/')))[-1] == year \
            else False


    print(
        len(ecom[ecom['CC Exp Date'].apply(lambda x: expires_in_year(x, 25))])
        , end='\n\n'
    )

    # or

    print(
        sum(ecom['CC Exp Date'].apply(lambda x: x[3:] == '25'))
        , end='\n\n'
    )

    # Display the Top 5 Most Popular Email Providers
    ecom['Email Provider'] = ecom['Email'].apply(lambda x: x.split('@')[-1])

    print(ecom['Email Provider'].value_counts().head(5), end='\n\n')
