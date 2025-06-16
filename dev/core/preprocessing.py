import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class PreProcessor:
    
    def __init__(self, param):
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def to_lowercase(self, tokens: list) -> list:
        return [token.lower() for token in tokens]
    
    def remove_stopwords(self, tokens: list) -> list:
        return [token  for token in tokens if token not in self.stopwords]
    
    def lemmatize(self, tokens: list) -> list:
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess_text(self, query: str) -> list:
        tokens = word_tokenize(query)  
        tokens = self.to_lowercase(tokens)
        tokens = self.remove_stopwords(tokens)
        return self.lemmatize(tokens)
        
    