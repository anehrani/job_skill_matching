#

#
import numpy as np
import pandas as pd
from gensim import models

from gensim.corpora import Dictionary, MmCorpus
from gensim import similarities

from dataStream import CSVReader, CSVWriter, DataStreamer
from utils import preprocess_text


def compare_similarity_of_texts( jobdescription_path , max_rows):
    """
    this function streams data (memory friendly) calculates the similarity matrix and write
    results to a csv file

    NOTE: This process can be easily paralelized, and it is better to do so otherwise it will be too slow

    input: folder path to read and write data

    output: nothing
    """
    # Load the corpus from the Matrix Market format
    job_descriptions_dict = Dictionary.load( 'data/job_descriptions_dict.dict')
    index = similarities.MatrixSimilarity.load( 'data/similarity_index.index')
    lsi_model = models.LsiModel.load(  "data/all_data_lsi_model.lsimodel" )

    data_streamer = DataStreamer( jobdescription_path, max_rows )

    # load job ids -> required in qriting
    job_ids = np.load('data/job_ids.npy', allow_pickle=True)


    csvwriter = CSVWriter (job_ids, "results/similarity_scores.csv", ["Job ID A", "Job ID B", "Similarity Score"] )

    del job_ids

    # Note: to keep csv files small (in case of large data, lets create a csv for every job ids)
    # this may cause to creation of large amout of files
    for row in data_streamer:
        #
        if len(row[0]) < 10: continue

        #
        processed_row = preprocess_text( row[3] )
        SimMat = index[ lsi_model[job_descriptions_dict.doc2bow( processed_row )]]


        csvwriter.write_similarity_to_csv( SimMat, row[0])



    csvwriter.file.close()







def prepare_similarity_data(jobdescription_path, path_to_save="data/"):
        """
        this part can improve by implementing clustering techniques in case of dealing with Big data
        to refuse memory problems
        (this is naive implementation only for small data )
        :param data_path:
        :param path_to_save:
        :return:
        """
        # load files
        df = pd.read_csv( jobdescription_path ).iloc[:50] # just to make it run faster I limited to 100 data

        #(remove null data)
        df.dropna(subset=['Job Description'], inplace=True)
        #  preprocess text
        job_descriptions = df['Job Description'].apply(preprocess_text)

        # save job ids to use later
        np.save(path_to_save + 'job_ids.npy', np.array(df["Job Id"], dtype=object), allow_pickle=True)

        # df is not required anymore, lets remove it and free up memory
        del df
        # save processed data
        np.save(path_to_save + 'job_descriptions.npy', np.array(job_descriptions, dtype=object), allow_pickle=True)

        # create dictionary and save
        job_descriptions_dict = Dictionary(job_descriptions)
        job_descriptions_dict.save(path_to_save + 'job_descriptions_dict.dict')

        # Convert the job descriptions into a bag-of-words format
        job_descriptions_corpus = [job_descriptions_dict.doc2bow(doc) for doc in job_descriptions]

        # Load pre-trained word2vec model
        # model = api.load("word2vec-google-news-300") # this model is large skip it
        # model = KeyedVectors.load_word2vec_format('../models/GoogleNews-vectors-negative300-SLIM.bin.gz', binary=True)

        lsi = models.LsiModel(job_descriptions_corpus, id2word=job_descriptions_dict,  num_topics=len(job_descriptions_corpus))

        lsi.save( path_to_save + "all_data_lsi_model.lsimodel")

        del job_descriptions_dict

        index = similarities.MatrixSimilarity(lsi[job_descriptions_corpus])

        # Save the corpus in the Matrix Market format
        MmCorpus.serialize(path_to_save + 'job_descriptions_corpus.mm', job_descriptions_corpus)

        del job_descriptions_corpus

        index.save(path_to_save + 'similarity_index.index')


def measure_similarity( jobdescription_path, max_rows ):
    prepare_similarity_data( jobdescription_path )

    # build the similarity matrix here
    compare_similarity_of_texts( jobdescription_path, max_rows )


if __name__ == "__main__":
    print("test ...")
    #
    # run this function only once!
    jobdescription_path = ""
    prepare_similarity_data(jobdescription_path)


    # build the similarity matrix here
    compare_similarity_of_texts(jobdescription_path)















