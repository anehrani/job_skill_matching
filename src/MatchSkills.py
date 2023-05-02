#
"""
    Trying to use LLMs to match skills

"""
#

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from utils import preprocess_text
from dataStream import DataStreamer, CSVWriter


class comparingModel:
    def __init__(self, model_path = ""):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        # model does not support sizes bigger than 512
        # todo: change model or use a tokenizer with size 512
        self.model = AutoModel.from_pretrained('bert-base-uncased')


    # Define function to match skills to job description using BERT
    def match_skills(self, job_description, skills_to_match):
        """
        This method is very slow but more accurate
        :param job_description:
        :param skills_to_match:
        :return:
                matched skills:

        """
        # Tokenize the job description - this should create a token of fized size 512
        # naivly cut or add zeros

        tokens = self.tokenizer.tokenize(job_description)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # Convert token to vocabulary indices
        tokens_tensor = torch.tensor([indexed_tokens])
        if len( tokens) > 512:
            tokens_tensor = torch.tensor([indexed_tokens[:512]])
        elif len( tokens) < 512:
            tokens_tensor = torch.tensor([indexed_tokens + [0] * (512 - len( tokens) ) ])
        segments_tensor = torch.tensor([[0] * 512])

        # Obtain the hidden states of the last layer
        with torch.no_grad():
            try:
                hidden_states = self.model(tokens_tensor, segments_tensor)
            except Exception as e:
                print(e)

        # Match skills to the job description
        matched_skills = []
        for skill in skills_to_match:
            skill_tokens = self.tokenizer.tokenize(skill)

            indexed_skill_tokens = self.tokenizer.convert_tokens_to_ids(skill_tokens)

            skill_tensor = torch.tensor([indexed_skill_tokens])
            with torch.no_grad():
                skill_outputs = self.model(skill_tensor)

            skill_outputs = torch.mean(skill_outputs[0], dim=1)
            try:
                similarities = torch.cosine_similarity(hidden_states[0], skill_outputs, dim=-1)
            except Exception as e:
                print(e)
            if similarities.max().numpy()> 0.45:
                matched_skills.append(skill)

        return matched_skills


def extractEntity():
    copmare_engin = comparingModel()


    jobdescription_stream = DataStreamer("../data/jobpostings.csv")
    # we can combine all skills and keepem in a single file to ease sreaming
    skill_stream = DataStreamer("../data/Skills.xlsx")

    # preprocess text
    # df.dropna(subset=['Job Description'], inplace=True)
    # load job ids -> required in qriting
    job_ids = np.load('../data/job_ids.npy', allow_pickle=True)


    csvwriter = CSVWriter (job_ids, "../results/JobEntities.csv", ["Job ID", "Entity"] )

    del job_ids

    # stream data and compare and arite results
    for j, jobdesc in enumerate(jobdescription_stream):
        if len(jobdesc[0]) < 10: continue # avoid headers

        extracted_entity = []

        for i, skill in enumerate(skill_stream):
            #
            if len(skill[3])<2: continue # skip null and one letter skills

            processed_skill =  preprocess_text(skill[3])
            processed_jobdesc =  " ".join( preprocess_text(jobdesc[3]))
            entity = copmare_engin.match_skills( processed_jobdesc, processed_skill )
            if not entity:
                continue
            extracted_entity.append( entity )

            # just to check code working, number of data to be checked is limited to 100
            if i>10: break # removing the break could cause running program for a long time


        # write the entity in csv file (if there sxists any)
        csvwriter.write_entity_to_csv(extracted_entity, jobdesc[0])

        # just to check code working, number of data to be checked is limited to 100
        if j>10: break

    csvwriter.file.close()





if __name__ == "__main__":

    print("testing")


    extractEntity()











print("finished!")

