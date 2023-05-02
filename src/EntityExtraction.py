#

#

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import spacy
from spacy.matcher import PhraseMatcher
import csv
import re
import string
from flair.data import Sentence
from flair.models import SequenceTagger
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')



# preprocessing texts
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and punctuation
    text = re.sub('[^a-zA-Z \n\.]', '', text)

    # Tokenize into words
    words = word_tokenize(text)

    # Remove stop words
    words = [word for word in words if word not in stopwords.words('english')]
    # remove punctuation
    clean_words = [token.translate(str.maketrans('', '', string.punctuation)) for token in words]


    # Stem words
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in clean_words]

    return ' '.join(words)

def extract_entities(text, skills_df, tech_skills_df):
    # Combine the skills and technology skills into a single list of keywords
    keywords = list(set(skills_df['Element Name']) | set(tech_skills_df['Example']))

    # Search for each keyword in the text and return a list of matches
    entities = []
    # Fuzzy match skills
    for token in keywords:
        # Extract entities from text using fuzzy string matching
        token = token.strip().split(" ")
        extracted_entities = process.extract(text, token, scorer=fuzz.token_sort_ratio)
        for item in extracted_entities:
            if item[1] > 10:
                entities.append(item[0])

        """
            # in case of exact matching (use PhraseMatcher) skipped: Create a PhraseMatcher object
            matcher = PhraseMatcher(nlp.vocab)
            doc = nlp(text)
            # Extract entities recognized by spaCy's NER
            # Convert entities to spaCy's Doc format and add them to the matcher
            patterns = [nlp.make_doc(entity) for entity in token]
            matcher.add('Entities', patterns)
            # Extract entities from text using the PhraseMatcher
            matches = matcher(doc)
            extracted_entities = [doc[start:end].text for match_id, start, end in matches]
        """


    # Remove duplicates
    entities = list(set(entities))

    return entities



def categorize_entities(entities, skills_df, tech_skills_df):
    skills = []
    tech_skills = []

    # Iterate over each entity and check if it appears in the soft or tech skills taxonomy data
    for entity in entities:
        if entity in list(skills_df['Element Name']):
            skills.append(entity)
        elif entity in list( tech_skills_df):
            tech_skills.append( entity)


def write_entity_extraction_results(Job_EntityValues):
    # write results as csv for later application
    with open('../data/Job_Entity.csv', 'w') as f:
        fieldnames = ["Job Id", "Entity"]
        writer = csv.DictWriter(f, fieldnames=fieldnames )
        writer.writeheader()
        for k, v in Job_EntityValues.items():
                writer.writerow( {"Job Id": k, "Entity": v})







if __name__ == "__main__":
    print("initialized ...")
    # Load the pre-trained English language model
    nlp = spacy.load('en_core_web_sm')

    # load files
    df = pd.read_csv('../data/jobpostings.csv').iloc[:100]

    # Load the skills and technology skills taxonomies into dataframes
    skills_df = pd.read_excel('../data/Skills.xlsx').iloc[:100]
    tech_skills_df = pd.read_excel('../data/Technology_Skills.xlsx').iloc[:100]

    # preprocess text
    df.dropna(subset=['Job Description'], inplace=True)
    df['Job Description'] = df['Job Description'].apply(preprocess_text)

    # Apply the entity extraction function to the job descriptions
    df['Entities'] = df['Job Description'].apply(lambda x: extract_entities(x, skills_df, tech_skills_df))



    # required entity info
    Job_EntityValues = {}

    for index, dfrow in df.iterrows():
        Job_EntityValues[dfrow["Job Id"]] = dfrow["Entities"]



    write_entity_extraction_results(Job_EntityValues)


    # Part Two similarity Score between Jobs











print("Finished ... ")
