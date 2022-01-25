import re
import pandas as pd

if __name__ == '__main__':
    # Pandas setting(s)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    # Read Input Data
    sal = pd.read_csv('data/Salaries.csv')

    # Display the head of the DataFrame
    print(sal.head(), end='\n\n')

    # Display some information about our DataFrame
    print(sal.info(), end='\n\n')

    # Display the average Base Payment
    print(sal['BasePay'].mean(), end='\n\n')

    # Display the highest Overtime Payment
    print(sal['OvertimePay'].max(), end='\n\n')

    # Display Job Title of a specific employee
    print(sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle'], end='\n\n')

    # Display Total Payment with Benefits for specific employee
    print(
        sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits'],
        end='\n\n'
    )

    # Display the highest paid employee including benefits
    print(
        sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()],
        end='\n\n'
    )

    # or

    print(
        sal.loc[sal['TotalPayBenefits'].idxmax()],
        end='\n\n'
    )

    # Display the lowest paid employee including benefits
    # ( For some reason the payment is negative number )
    print(
        sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].min()],
        end='\n\n'
    )

    # or

    print(
        sal.loc[sal['TotalPayBenefits'].idxmin()],
        end='\n\n'
    )

    # Display the average Base Payment
    print(sal.groupby(by='Year').mean()['BasePay'], end='\n\n')

    # Display the number of unique job titles
    print(sal['JobTitle'].nunique(), end='\n\n')

    # Display the Top 5 Most Common Jobs
    print(sal['JobTitle'].value_counts().head(5), end='\n\n')

    # Display the number of Jobs that were represented by only one person in
    # 2013
    print(
        len(
            sal[sal['Year'] == 2013]['JobTitle'].value_counts().loc[
                lambda x: x == 1
            ]
        ),
        end='\n\n'
    )

    # or

    print(
        sum(sal[sal['Year'] == 2013]['JobTitle'].value_counts() == 1),
        end='\n\n'
    )


    def chief_string(title: str) -> bool:
        """Returns boolean value based on if 'chief' is in Job Title

        Args:
            title (str): Job Title

        Returns:
            bool: Job Title contains the word 'chief'
        """

        return \
            True if 'chief' in list(re.split('[, ]', title.lower())) else False


    # Display the number of Jobs that contain the word 'Chief'
    print(sum(sal['JobTitle'].apply(lambda x: chief_string(x))), end='\n\n')

    # Display the correlation between the Job Title Length and the Total
    # Payment
    sal['title_len'] = sal['JobTitle'].apply(len)

    print(sal[['title_len', 'TotalPayBenefits']].corr(), end='\n\n')
