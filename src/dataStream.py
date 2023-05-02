#
#


import csv
from smart_open import open
import pandas as pd

class CSVReader:
    def __init__(self, file_path):
        self.file = open(file_path, 'r')
        self.reader = csv.reader(self.file)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.reader)


class DataStreamer:
    def __init__(self, file_path, max_rows = 10):

        self.max_rows = max_rows
        self.current_row = 0
        if file_path.split(".")[-1] == "csv":
            self.file = open(file_path, 'r')
            self.reader = csv.reader(self.file)
            self.ftype = "csv"
        elif file_path.split(".")[-1] == "xlsx":

            self.file = open(file_path, 'rb')
            self.reader = iter(pd.read_excel(self.file).values)
            self.ftype = "xlsx"
        else:
            raise Exception("Can't read the data file.")

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_row >= self.max_rows:
            self.file.close()
            raise StopIteration
        self.current_row +=1

        if self.ftype == "csv":
            return next(self.reader)
        elif self.ftype == "xlsx":
            return self.reader.__next__()






class CSVWriter():
    def __init__(self, job_ids, file_path, header ):
        self.job_ids = job_ids
        self.file_path = file_path
        self.file = open(self.file_path, 'w')
        fieldnames = header
        self.writer = csv.DictWriter( self.file, fieldnames=fieldnames )
        self.writer.writeheader()

    def write_similarity_to_csv(self, sim_matrix, jid ):
        for i in range(sim_matrix.shape[0]):
           self.writer.writerow( {"Job ID A": jid , "Job ID B": self.job_ids[i],  "Similarity Score": sim_matrix[i]})

    def write_entity_to_csv(self, entities, jid ):
        entity_to_write = []
        for entity in entities:
            entity_to_write += entity

        entity_to_write = " ".join(entity_to_write)
        self.writer.writerow( {"Job ID": jid , "Entity": entity_to_write })





