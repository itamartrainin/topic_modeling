#%%
import pandas as pd
from datetime import datetime
import nltk
from tqdm import tqdm

tqdm.pandas()

# %%
df = pd.read_excel('C:/Data/TopicModeling/realDonaldTrump_tweets.xlsx', names=['date', 'tag', 'content', 'uri'])
print('Total rows: ', len(df))

# %%
df['datetime'] = df['date'].apply(lambda d: datetime.strptime(d, '%B %d, %Y at %I:%M%p'))
df['year'] = df['datetime'].apply(lambda d: d.year)
df['month'] = df['datetime'].apply(lambda d: d.strftime('%b'))
df['day'] = df['datetime'].apply(lambda d: d.day)

# %%
print(df['year'].value_counts())
print(df['month'].value_counts())
print(df['day'].value_counts())

# %%
df['tokenized'] = df['content'].progress_apply(lambda x: nltk.word_tokenize(x))
df['pos'] = df['tokenized'].progress_apply(lambda x: nltk.pos_tag(x))

# %%
sentence = df['pos'][0]
grammar = "NP: {<DT>?<JJ>*<NN>}"
cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
print(result)