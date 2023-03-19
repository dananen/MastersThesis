from bs4 import BeautifulSoup
import requests
import pandas as pd
from MainAPIHelperFunctions import string_to_datetime

######### INPUT #############
reverted_data = True
#############################

if reverted_data:
    file = '/Users/dansvenonius/Desktop/Preprocessing output/Reverted Data/user_data.csv'
else:
    file = '/Users/dansvenonius/Desktop/Preprocessing output/Not Reverted Data/user_data.csv'


user_df = pd.read_csv(file)
user_df['acc_created'] = ''

for index, row in user_df.iterrows():
    if index%100 == 0:
        print(index)
    soup = BeautifulSoup(requests.get("https://api.openstreetmap.org/api/0.6/user/" + str(row['uid'])).content, 'lxml-xml')
    soup = soup.find('user')
    if soup != None and 'account_created' in soup.attrs.keys():
        user_df.loc[index, 'acc_created'] = string_to_datetime(soup.attrs['account_created'], iso=True)

user_df.to_csv(file, index=False)