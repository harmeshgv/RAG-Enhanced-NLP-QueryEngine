python -m venv myenv

.\myenv\Scripts\activate

pip install -r requirements.txt


pip install openpyxl



```python
import pandas as pd
import numpy as np

import faiss

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from tqdm import tqdm
import time

import ipywidgets as widgets
widgets.IntSlider()

query= "what is the max rating given in home and lifestyle?"

```


```python
df = pd.read_csv("output_table.csv")
```


```python
model = SentenceTransformer('all-MiniLM-L6-v2')
```


```python
column_names = df.columns.to_list()
column_name_embeddings = model.encode(column_names)
```


```python
column_index_mapping = {
    "invoice id": 0,
    "city": 1,
    "gender": 2,
    "product line": 3,
    "unit price": 4,
    "quantity": 5,
    "total": 6,
    "date": 7,
    "payment": 8,
    "gross income": 9,
    "rating": 10
}

```


```python
column_name_mapping = {
    'city' : df["city"].unique().tolist(),
    'gender' : df["gender"].unique().tolist(),
    'product line': df["product line"].unique().tolist(),
    'payment' : df["payment"].unique().tolist()
    
}
```


```python
column_name_mapping
```




    {'city': ['Yangon', 'Naypyitaw', 'Mandalay'],
     'gender': ['Female', 'Male'],
     'product line': ['Health beauty',
      'Electronic accessories',
      'Home lifestyle',
      'Sports travel',
      'Food beverages',
      'Fashion accessories'],
     'payment': ['Ewallet', 'Cash', 'Credit card']}




```python
column_name_embeddings
```




    array([[-0.1117169 ,  0.11272218, -0.02423377, ...,  0.00350045,
             0.03952534, -0.05215904],
           [ 0.05124492,  0.07561278, -0.03347927, ...,  0.02809969,
            -0.03862095,  0.05421344],
           [ 0.02549297,  0.05705191, -0.0425216 , ...,  0.01834778,
             0.07324108, -0.05377625],
           ...,
           [-0.04512803,  0.09326787, -0.01289233, ..., -0.03927517,
             0.05541484, -0.10761316],
           [ 0.02171222,  0.0022199 ,  0.00923696, ..., -0.05872287,
             0.03283434, -0.12940843],
           [-0.06318381, -0.00255952, -0.12002307, ..., -0.06058924,
            -0.00833742,  0.05499737]], dtype=float32)




```python
def preprocess_query(user_query):
    # Filter out stop words from the user query
    filtered_text = [word for word in user_query.split() if word.lower() not in ENGLISH_STOP_WORDS]
    
    # Lemmatize each word in the filtered text to get its base form
    lemmetized_text = [WordNetLemmatizer().lemmatize(word) for word in filtered_text]
    
    # Return the preprocessed list of words
    return lemmetized_text

```


```python
#column filtering
def get_matching_columns(list_query, threshold=0.7):
    matched_columns = []

    for query in list_query:
        query_embedding = model.encode([query])
        
        # Check similarity with column names
        similarities = cosine_similarity(query_embedding, column_name_embeddings)
        top_column_indices = np.argsort(similarities[0])[::-1]
        
        added = False
        for idx in top_column_indices:
            if similarities[0][idx] >= threshold:
                matched_columns.append(column_names[idx])
                added = True
                break
        
        # Fallback: Check similarity with column values using embeddings
        if not added:
            for column, values in column_name_mapping.items():
                value_embeddings = model.encode(values)
                value_similarities = cosine_similarity(query_embedding, value_embeddings)
                
                # Check if any value matches the threshold
                max_similarity = value_similarities.max()
                if max_similarity >= threshold:
                    matched_columns.append(column)
                    break

    # Deduplicate the column list
    return list(set(matched_columns))
```


```python
final_query = preprocess_query(query) # Example query
```


```python
final_query  # Example query after preprocessing
```




    ['max', 'rating', 'given', 'home', 'lifestyle?']




```python
get_matching_columns(final_query) # Queries that match the columns 
```




    ['rating', 'product line']




```python
QA = pd.read_excel("QA_dataset_share.xlsx")  # Excel Q/A data provided for testing
```


```python
QA.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>question</th>
      <th>row index</th>
      <th>column index</th>
      <th>answer</th>
      <th>filtered row index</th>
      <th>filtered column index</th>
      <th>generated response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What product line is in the latest entry?</td>
      <td>999</td>
      <td>3</td>
      <td>Fashion accessories</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>On what date did the first transaction occur?</td>
      <td>17, 245, 450, 484, 496, 523, 567, 696, 829, 83...</td>
      <td>7</td>
      <td>1/1/2019</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What is the latest transaction date?</td>
      <td>158, 306, 473, 474, 643, 646, 671, 881, 883, 9...</td>
      <td>7</td>
      <td>3/30/2019</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>what is the max rating given in home and lifes...</td>
      <td>2, 7, 19, 22, 25, 39, 40, 41, 54, 56, 58, 61, ...</td>
      <td>3,10</td>
      <td>9.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>How many transactions involved Male customers ...</td>
      <td>331, 464, 540, 708, 710</td>
      <td>2,10</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
QA.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 70 entries, 0 to 69
    Data columns (total 7 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   question               70 non-null     object 
     1   row index              70 non-null     object 
     2   column index           70 non-null     object 
     3   answer                 70 non-null     object 
     4   filtered row index     0 non-null      float64
     5   filtered column index  0 non-null      float64
     6   generated response     0 non-null      float64
    dtypes: float64(3), object(4)
    memory usage: 4.0+ KB
    


```python
def update_filtered_column_indices(df, column_name_mapping):
    """
    Updates the DataFrame by adding a 'filtered column index' column
    based on mapped question-to-column indices.

    Parameters:
    df (pd.DataFrame): Input DataFrame with a 'question' column.
    column_name_mapping (dict): Dictionary mapping column names to their indices.

    Returns:
    pd.DataFrame: DataFrame with an updated 'filtered column index' column.
    """
    
    def map_question_to_indices(question):
        """
        Maps a given question to column indices based on matching columns.

        Parameters:
        question (str): A question string to be processed.

        Returns:
        list: List of column indices corresponding to matching columns.
        """
        # Preprocess the question to normalize or clean it
        final_query = preprocess_query(question)
        
        # Retrieve column names that match the processed query
        matched_columns = get_matching_columns(final_query)
        
        # Map the matched columns to their corresponding indices
        return [column_name_mapping[col] for col in matched_columns if col in column_name_mapping]
    
    # Enable the progress bar for DataFrame operations
    tqdm.pandas(desc="Processing questions")
    
    # Apply the mapping function to each question in the DataFrame and store the result in a new column
    df["filtered column index"] = df["question"].progress_apply(lambda q: map_question_to_indices(q))
    
    # Return the updated DataFrame
    return df

```


```python
QA = update_filtered_column_indices(QA, column_index_mapping)
print(QA)

```

    Processing questions: 100%|██████████| 70/70 [00:13<00:00,  5.00it/s]

                                                 question  \
    0           What product line is in the latest entry?   
    1       On what date did the first transaction occur?   
    2                What is the latest transaction date?   
    3   what is the max rating given in home and lifes...   
    4   How many transactions involved Male customers ...   
    ..                                                ...   
    65                          What is the minimum cost?   
    66  How many transactions involved Ewallet payment...   
    67  What is the total gross income for transaction...   
    68  How many transactions involved Female customer...   
    69  What is the total gross income for transaction...   
    
                                                row index column index  \
    0                                                 999            3   
    1   17, 245, 450, 484, 496, 523, 567, 696, 829, 83...            7   
    2   158, 306, 473, 474, 643, 646, 671, 881, 883, 9...            7   
    3   2, 7, 19, 22, 25, 39, 40, 41, 54, 56, 58, 61, ...         3,10   
    4                             331, 464, 540, 708, 710         2,10   
    ..                                                ...          ...   
    65  0,\n  1,\n  2,\n  3,\n  4,\n  5,\n  6,\n  7,\n...            6   
    66  12,\n  20,\n  23,\n  36,\n  116,\n  157,\n  17...          5,8   
    67                         7, 362, 582, 656, 699, 771       1,9,10   
    68  1, 12, 28, 51, 79, 126, 136, 199, 216, 237, 23...         2, 5   
    69           1, 28, 306, 376, 403, 449, 499, 599, 700       2,9,10   
    
                     answer  filtered row index filtered column index  \
    0   Fashion accessories                 NaN                   [3]   
    1              1/1/2019                 NaN                   [7]   
    2             3/30/2019                 NaN                   [7]   
    3                   9.9                 NaN               [10, 3]   
    4                     5                 NaN               [10, 2]   
    ..                  ...                 ...                   ...   
    65                10.67                 NaN                    []   
    66                   36                 NaN                [8, 5]   
    67               139.84                 NaN         [10, 9, 1, 6]   
    68                   56                 NaN                [5, 2]   
    69                119.6                 NaN         [10, 9, 2, 6]   
    
        generated response  
    0                  NaN  
    1                  NaN  
    2                  NaN  
    3                  NaN  
    4                  NaN  
    ..                 ...  
    65                 NaN  
    66                 NaN  
    67                 NaN  
    68                 NaN  
    69                 NaN  
    
    [70 rows x 7 columns]
    

    
    


```python
QA
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>question</th>
      <th>row index</th>
      <th>column index</th>
      <th>answer</th>
      <th>filtered row index</th>
      <th>filtered column index</th>
      <th>generated response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What product line is in the latest entry?</td>
      <td>999</td>
      <td>3</td>
      <td>Fashion accessories</td>
      <td>NaN</td>
      <td>[3]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>On what date did the first transaction occur?</td>
      <td>17, 245, 450, 484, 496, 523, 567, 696, 829, 83...</td>
      <td>7</td>
      <td>1/1/2019</td>
      <td>NaN</td>
      <td>[7]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What is the latest transaction date?</td>
      <td>158, 306, 473, 474, 643, 646, 671, 881, 883, 9...</td>
      <td>7</td>
      <td>3/30/2019</td>
      <td>NaN</td>
      <td>[7]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>what is the max rating given in home and lifes...</td>
      <td>2, 7, 19, 22, 25, 39, 40, 41, 54, 56, 58, 61, ...</td>
      <td>3,10</td>
      <td>9.9</td>
      <td>NaN</td>
      <td>[10, 3]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>How many transactions involved Male customers ...</td>
      <td>331, 464, 540, 708, 710</td>
      <td>2,10</td>
      <td>5</td>
      <td>NaN</td>
      <td>[10, 2]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>65</th>
      <td>What is the minimum cost?</td>
      <td>0,\n  1,\n  2,\n  3,\n  4,\n  5,\n  6,\n  7,\n...</td>
      <td>6</td>
      <td>10.67</td>
      <td>NaN</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>66</th>
      <td>How many transactions involved Ewallet payment...</td>
      <td>12,\n  20,\n  23,\n  36,\n  116,\n  157,\n  17...</td>
      <td>5,8</td>
      <td>36</td>
      <td>NaN</td>
      <td>[8, 5]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>67</th>
      <td>What is the total gross income for transaction...</td>
      <td>7, 362, 582, 656, 699, 771</td>
      <td>1,9,10</td>
      <td>139.84</td>
      <td>NaN</td>
      <td>[10, 9, 1, 6]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>68</th>
      <td>How many transactions involved Female customer...</td>
      <td>1, 12, 28, 51, 79, 126, 136, 199, 216, 237, 23...</td>
      <td>2, 5</td>
      <td>56</td>
      <td>NaN</td>
      <td>[5, 2]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>69</th>
      <td>What is the total gross income for transaction...</td>
      <td>1, 28, 306, 376, 403, 449, 499, 599, 700</td>
      <td>2,9,10</td>
      <td>119.6</td>
      <td>NaN</td>
      <td>[10, 9, 2, 6]</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>70 rows × 7 columns</p>
</div>




```python
QA.to_excel("updated_QA.xlsx", index=False)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 11 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   invoice id    1000 non-null   object 
     1   city          1000 non-null   object 
     2   gender        1000 non-null   object 
     3   product line  1000 non-null   object 
     4   unit price    1000 non-null   float64
     5   quantity      1000 non-null   int64  
     6   total         1000 non-null   float64
     7   date          1000 non-null   object 
     8   payment       1000 non-null   object 
     9   gross income  1000 non-null   float64
     10  rating        1000 non-null   float64
    dtypes: float64(4), int64(1), object(6)
    memory usage: 86.1+ KB
    

### Row filtering


```python
from typing import List
```


```python
updated_qa = pd.read_excel("updated_QA.xlsx")
updated_qa.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>question</th>
      <th>row index</th>
      <th>column index</th>
      <th>answer</th>
      <th>filtered row index</th>
      <th>filtered column index</th>
      <th>generated response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What product line is in the latest entry?</td>
      <td>999</td>
      <td>3</td>
      <td>Fashion accessories</td>
      <td>NaN</td>
      <td>[3]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>On what date did the first transaction occur?</td>
      <td>17, 245, 450, 484, 496, 523, 567, 696, 829, 83...</td>
      <td>7</td>
      <td>1/1/2019</td>
      <td>NaN</td>
      <td>[7]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What is the latest transaction date?</td>
      <td>158, 306, 473, 474, 643, 646, 671, 881, 883, 9...</td>
      <td>7</td>
      <td>3/30/2019</td>
      <td>NaN</td>
      <td>[7]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>what is the max rating given in home and lifes...</td>
      <td>2, 7, 19, 22, 25, 39, 40, 41, 54, 56, 58, 61, ...</td>
      <td>3,10</td>
      <td>9.9</td>
      <td>NaN</td>
      <td>[10, 3]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>How many transactions involved Male customers ...</td>
      <td>331, 464, 540, 708, 710</td>
      <td>2,10</td>
      <td>5</td>
      <td>NaN</td>
      <td>[10, 2]</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
def row_filtering():
    lst = updated_qa["filtered column index"]
    for clms in lst:
        
    
```


```python
column_data = read_excel_column(r"C:\Users\harme\Downloads\Hackathon - LPU_2025\TQ-29\updated_QA.xlsx", )
```

#### working on filtering for a single query


```python
embedded_data = df["product line"].apply(lambda x: model.encode([x])[0])

```


```python
embedded_data = np.array(embedded_data.tolist())
```


```python
ratings = df["rating"].values.reshape(-1, 1)  # Reshape for concatenation

```


```python
embedded_data = np.array(embedded_data.tolist())

```


```python
ratings = df["rating"].values.reshape(-1, 1)  # Reshape for concatenation

```


```python
vector_data = np.hstack((embedded_data, ratings))

```


```python
# Determine the dimension for FAISS indexing
dimension = vector_data.shape[1]
```


```python

# Create a FAISS index for L2-based similarity search
index = faiss.IndexFlatL2(dimension)

# Add vectors to the FAISS index
index.add(vector_data)

# Save the FAISS index for future use
faiss.write_index(index, "vector_store.index")
```


```python
# List of query words
query_words = ["home", "lifestyle", "max", "given"]

# Initialize a set to store unique matching row indices
matching_row_indices = set()

for word in query_words:
    # Generate query embedding for the input word
    query_embedding = model.encode([word])[0]

    # Append a dummy rating value for comparison (matching dimensions)
    query_vector = np.append(query_embedding, [0.0]).reshape(1, -1)

    # Perform similarity search for top 200 matches
    D, I = index.search(query_vector, k=200)
    print(f"Word: {word}, Distances: {D}, Indexes: {I}")

    # Filter rows based on the threshold
    filtered_indices = [i for i, distance in enumerate(D[0]) if distance <= threshold]

    # Collect row indices for valid matches
    if filtered_indices:
        matching_row_indices.update(I[0][filtered_indices])

# Convert to a sorted list for better handling
matching_row_indices = sorted(list(matching_row_indices))
print("Matching row indices:", matching_row_indices)

# Retrieve the full matching rows from the DataFrame
matching_rows = df.iloc[matching_row_indices]
print("Matching rows:", matching_rows)



```

    Word: home, Distances: [[17.406937 17.406937 17.406937 17.406937 17.522652 17.527887 17.527887
      17.527887 17.527887 17.556938 17.556938 17.556938 17.60545  17.60545
      17.674591 18.236935 18.236935 18.236935 18.236935 18.332651 18.332651
      18.332651 18.332651 18.337887 18.366938 18.415451 18.415451 18.415451
      18.415451 18.484589 18.484589 18.484589 19.086939 19.086939 19.086939
      19.086939 19.086939 19.16265  19.16265  19.16265  19.16265  19.16265
      19.167885 19.167885 19.167885 19.167885 19.196936 19.196936 19.196936
      19.196936 19.196936 19.24545  19.24545  19.314587 19.314587 19.956938
      19.956938 19.956938 19.956938 20.012653 20.012653 20.017889 20.017889
      20.04694  20.04694  20.04694  20.04694  20.095451 20.164593 20.164593
      20.164593 20.164593 20.846937 20.846937 20.846937 20.882652 20.887888
      20.887888 20.887888 20.887888 20.916939 20.916939 20.96545  20.96545
      20.96545  21.034592 21.034592 21.034592 21.756937 21.772652 21.772652
      21.772652 21.772652 21.772652 21.777887 21.777887 21.777887 21.777887
      21.85545  21.85545  21.85545  21.85545  21.924591 22.682652 22.686935
      22.686935 22.686935 22.686935 22.687887 22.687887 22.716938 22.716938
      22.76545  22.76545  23.636938 23.636938 23.646936 23.69545  23.764587
      23.764587 23.764587 23.764587 23.764587 23.764587 24.562653 24.567888
      24.567888 24.59694  24.59694  24.606937 24.606937 24.606937 24.606937
      24.64545  24.64545  24.64545  24.714592 24.714592 24.714592 25.532652
      25.537888 25.566938 25.566938 25.566938 25.566938 25.615452 25.615452
      25.615452 25.615452 25.615452 25.68459  25.68459  25.68459  26.522652
      26.522652 26.522652 26.522652 26.522652 26.527887 26.527887 26.527887
      26.556938 26.556938 26.556938 26.556938 26.556938 26.60545  26.60545
      26.60545  26.60545  26.60545  26.606936 26.606936 26.606936 26.606936
      26.606936 26.674591 26.674591 26.674591 27.53265  27.53265  27.53265
      27.53265  27.53265  27.537886 27.537886 27.566936 27.566936 27.566936
      27.566936 27.566936 27.615448 27.636934 27.636934 27.636934 27.68459
      27.68459  27.68459  28.562649 28.562649]], Indexes: [[119 307 821 998  85  72 618 643 848 226 328 672 790 876 379 243 289 805
      909 235 569 669 832  47 784 100 407 750 782   5  97 915 113 488 687 697
      862 122 263 476 548 615 108 249 383 614 776 895 902 938 944 849 855 381
      761  19  22 268 996 441 519 701 868  69 395 590 986 795 238 305 829 988
      125 298 757 213 178 302 312 561 589 668 623 670 730 314 477 617 123  15
       31 467 770 824 456 460 557 990  10 601 604 716 878 110 347 633 765 951
      396 899  16 473 191 675 166 353 301 947  37 292 303 308 454 641 593 452
      707 349 731 212 545 622 792  30 150 177  20 246 674 684 977 149 196 236
      867 239 371 404 538 659 102 317 756 482 542 839 842 885 634 815 859 313
      523 627 635 852 195 424 536 702 896 273 509 901 920 935 563 811 814  32
       92 653 721 837 539 932  21  33  96 410 858 208 253 372 922 348 801 978
      337 357]]
    Word: lifestyle, Distances: [[17.09705  17.09705  17.09705  17.160126 17.160126 17.160126 17.160126
      17.482584 17.482584 17.495176 17.640255 17.640255 17.640255 17.640255
      17.776726 17.90705  17.990124 17.990124 17.990124 17.990124 18.292583
      18.292583 18.292583 18.292583 18.305176 18.305176 18.305176 18.305176
      18.450254 18.586725 18.586725 18.586725 18.737047 18.737047 18.737047
      18.737047 18.737047 18.840128 18.840128 18.840128 18.840128 18.840128
      19.122581 19.122581 19.135174 19.135174 19.135174 19.135174 19.135174
      19.280252 19.280252 19.280252 19.280252 19.416723 19.416723 19.587051
      19.587051 19.587051 19.587051 19.710127 19.710127 19.710127 19.710127
      19.972586 19.985178 19.985178 20.130257 20.130257 20.266727 20.266727
      20.266727 20.266727 20.45705  20.45705  20.600126 20.600126 20.600126
      20.842585 20.842585 20.842585 20.855177 21.000256 21.000256 21.000256
      21.000256 21.136726 21.136726 21.136726 21.510126 21.732584 21.732584
      21.732584 21.732584 21.745176 21.745176 21.745176 21.745176 21.745176
      21.890255 21.890255 21.890255 21.890255 22.026726 22.25705  22.25705
      22.440125 22.440125 22.440125 22.440125 22.642584 22.642584 22.655176
      22.800255 22.800255 23.187048 23.390127 23.390127 23.572582 23.866724
      23.866724 23.866724 23.866724 23.866724 23.866724 24.13705  24.13705
      24.360126 24.360126 24.360126 24.360126 24.522585 24.522585 24.522585
      24.535177 24.680256 24.680256 24.816727 24.816727 24.816727 25.10705
      25.10705  25.10705  25.10705  25.492584 25.492584 25.492584 25.492584
      25.492584 25.505177 25.650255 25.786726 25.786726 25.786726 26.09705
      26.09705  26.09705  26.09705  26.09705  26.360125 26.360125 26.360125
      26.360125 26.360125 26.482584 26.482584 26.482584 26.482584 26.482584
      26.495176 26.495176 26.495176 26.495176 26.495176 26.640255 26.640255
      26.640255 26.776726 26.776726 26.776726 27.107048 27.107048 27.107048
      27.107048 27.107048 27.390123 27.390123 27.390123 27.492582 27.505175
      27.505175 27.505175 27.505175 27.505175 27.650253 27.650253 27.786724
      27.786724 27.786724 28.522581 28.522581]], Indexes: [[226 328 672 119 307 821 998 790 876  85  72 618 643 848 379 784 243 289
      805 909 100 407 750 782 235 569 669 832  47   5  97 915 776 895 902 938
      944 113 488 687 697 862 849 855 122 263 476 548 615 108 249 383 614 381
      761  69 395 590 986  19  22 268 996 795 441 519 701 868 238 305 829 988
      589 668 125 298 757 623 670 730 213 178 302 312 561 314 477 617 123  10
      601 604 716  15  31 467 770 824 456 460 557 990 878  16 473 347 633 765
      951 191 675 110 396 899 301 166 353 947  37 292 303 308 454 641 349 731
      212 545 622 792  30 150 177 593 452 707  20 246 674 149 196 236 867 239
      371 404 538 659 684 977 102 317 756 313 523 627 635 852 273 509 901 920
      935 195 424 536 702 896 482 542 839 842 885 634 815 859 563 811 814  21
       33  96 410 858 253 372 922 208  32  92 653 721 837 539 932 348 801 978
      117 606]]
    Word: max, Distances: [[17.674002 17.674002 17.679188 17.679188 17.679188 17.718435 17.774752
      17.85526  17.85526  17.85526  17.85526  18.484001 18.484001 18.484001
      18.484001 18.489187 18.528435 18.528435 18.528435 18.528435 18.584751
      18.584751 18.584751 18.619781 18.619781 18.619781 18.619781 18.66526
      19.314    19.314    19.319185 19.319185 19.319185 19.319185 19.319185
      19.358433 19.358433 19.358433 19.358433 19.358433 19.41475  19.41475
      19.44978  19.44978  19.44978  19.44978  19.495258 19.495258 19.495258
      19.495258 20.164003 20.16919  20.16919  20.16919  20.16919  20.208437
      20.208437 20.264753 20.264753 20.264753 20.264753 20.299784 20.299784
      20.299784 20.299784 20.299784 20.345263 20.345263 21.034002 21.034002
      21.034002 21.039188 21.039188 21.078436 21.134752 21.134752 21.134752
      21.169783 21.169783 21.169783 21.169783 21.215261 21.215261 21.215261
      21.215261 21.924002 21.924002 21.924002 21.924002 21.968435 21.968435
      21.968435 21.968435 21.968435 22.024752 22.059782 22.059782 22.059782
      22.10526  22.10526  22.10526  22.10526  22.834002 22.834002 22.839188
      22.839188 22.878435 22.969782 23.01526  23.01526  23.764    23.769186
      23.86475  23.86475  23.86475  23.86475  23.86475  23.86475  23.89978
      23.89978  23.89978  23.89978  24.714003 24.714003 24.714003 24.719189
      24.719189 24.758436 24.814753 24.814753 24.814753 24.849783 24.849783
      24.895262 24.895262 25.684002 25.684002 25.684002 25.684002 25.684002
      25.689188 25.689188 25.689188 25.689188 25.728436 25.784752 25.784752
      25.784752 25.819782 25.819782 25.819782 25.819782 25.865261 26.674002
      26.674002 26.674002 26.674002 26.674002 26.679188 26.679188 26.679188
      26.679188 26.679188 26.718435 26.718435 26.718435 26.718435 26.718435
      26.774752 26.774752 26.774752 26.85526  26.85526  26.85526  27.684
      27.689186 27.689186 27.689186 27.689186 27.689186 27.728434 27.728434
      27.728434 27.728434 27.728434 27.78475  27.78475  27.78475  27.81978
      27.81978  27.81978  27.81978  27.81978  27.86526  27.86526  28.713999
      28.713999 28.713999 28.713999 28.758432]], Indexes: [[790 876 226 328 672  85 379  72 618 643 848 100 407 750 782 784 235 569
      669 832   5  97 915 119 307 821 998  47 849 855 776 895 902 938 944 122
      263 476 548 615 381 761 243 289 805 909 108 249 383 614 795  69 395 590
      986 441 519 238 305 829 988 113 488 687 697 862 701 868 623 670 730 589
      668 213 314 477 617  19  22 268 996 178 302 312 561  10 601 604 716  15
       31 467 770 824 878 125 298 757 456 460 557 990 191 675  16 473 110 123
      396 899 947 301  37 292 303 308 454 641 347 633 765 951  30 150 177 349
      731 593  20 246 674 166 353 452 707 239 371 404 538 659 149 196 236 867
      684 102 317 756 212 545 622 792 977 195 424 536 702 896 313 523 627 635
      852 482 542 839 842 885 563 811 814 634 815 859 208  21  33  96 410 858
       32  92 653 721 837 348 801 978 273 509 901 920 935 539 932 117 606 608
      746 337]]
    Word: given, Distances: [[17.4857   17.4857   17.4857   17.626842 17.626842 17.640625 17.655954
      17.725723 17.725723 17.725723 17.725723 18.2957   18.427841 18.427841
      18.427841 18.427841 18.436842 18.436842 18.436842 18.436842 18.450624
      18.450624 18.450624 18.465954 18.465954 18.465954 18.465954 18.535723
      19.125698 19.125698 19.125698 19.125698 19.125698 19.25784  19.25784
      19.25784  19.25784  19.26684  19.26684  19.280622 19.280622 19.295952
      19.295952 19.295952 19.295952 19.295952 19.36572  19.36572  19.36572
      19.36572  19.975702 19.975702 19.975702 19.975702 20.107841 20.107841
      20.107841 20.107841 20.107841 20.116844 20.130627 20.130627 20.130627
      20.130627 20.145956 20.145956 20.215725 20.215725 20.845701 20.845701
      20.97784  20.97784  20.97784  20.97784  20.986843 20.986843 20.986843
      21.000626 21.000626 21.000626 21.015955 21.085724 21.085724 21.085724
      21.085724 21.86784  21.86784  21.86784  21.876842 21.876842 21.876842
      21.876842 21.890625 21.905954 21.905954 21.905954 21.905954 21.905954
      21.975723 21.975723 21.975723 21.975723 22.6457   22.6457   22.77784
      22.786842 22.786842 22.815954 22.885723 22.885723 23.575699 23.70784
      23.70784  23.70784  23.70784  23.71684  23.730623 23.730623 23.730623
      23.730623 23.730623 23.730623 24.525702 24.525702 24.65784  24.65784
      24.666843 24.666843 24.666843 24.680626 24.680626 24.680626 24.695955
      24.765724 24.765724 25.4957   25.4957   25.4957   25.4957   25.627842
      25.627842 25.627842 25.627842 25.636843 25.636843 25.636843 25.636843
      25.636843 25.650625 25.650625 25.650625 25.665955 25.735723 26.4857
      26.4857   26.4857   26.4857   26.4857   26.626842 26.626842 26.626842
      26.626842 26.626842 26.640625 26.640625 26.640625 26.655954 26.655954
      26.655954 26.655954 26.655954 26.725723 26.725723 26.725723 27.495699
      27.495699 27.495699 27.495699 27.495699 27.627838 27.627838 27.627838
      27.627838 27.627838 27.63684  27.650623 27.650623 27.650623 27.665953
      27.665953 27.665953 27.665953 27.665953 27.735722 27.735722 28.657837
      28.657837 28.657837 28.66684  28.66684 ]], Indexes: [[226 328 672 790 876 379  85  72 618 643 848 784 119 307 821 998 100 407
      750 782   5  97 915 235 569 669 832  47 776 895 902 938 944 243 289 805
      909 849 855 381 761 122 263 476 548 615 108 249 383 614  69 395 590 986
      113 488 687 697 862 795 238 305 829 988 441 519 701 868 589 668  19  22
      268 996 623 670 730 314 477 617 213 178 302 312 561 125 298 757  10 601
      604 716 878  15  31 467 770 824 456 460 557 990  16 473 123 191 675 110
      396 899 301 347 633 765 951 947  37 292 303 308 454 641 349 731 166 353
       30 150 177  20 246 674 593 452 707 149 196 236 867 212 545 622 792 239
      371 404 538 659 102 317 756 684 977 313 523 627 635 852 195 424 536 702
      896 563 811 814 482 542 839 842 885 634 815 859  21  33  96 410 858 273
      509 901 920 935 208 348 801 978  32  92 653 721 837 539 932 253 372 922
      117 606]]
    Matching row indices: []
    Matching rows: Empty DataFrame
    Columns: [invoice id, city, gender, product line, unit price, quantity, total, date, payment, gross income, rating]
    Index: []
    

For row filtering and retrieval we can create chunks based on rime casting 


```python

```
