
"""
    main function to run the entity extraction and similarity score generation

"""
import sys
sys.path.append('src/')
sys.path.append('data/')
sys.path.append('results/')

from src.MatchSkills import extractEntity
from src.MeasureSimilarity import measure_similarity


# import argparse
# args = argparse.ArgumentParser("taking the initial data path")
# args.add_argument("--jd", "-jobdesc", default="data/jobpostings.csv", required=False, help="path to the jobdescription file")
# args.add_argument("--skl", "-skill", default="data/Skills.xlsx", required=False, help="path to the skills file")
# args.add_argument("--tsk", "-teskil", default="data/Technology_Skills.xlsx", required=False, help="path to the techskills file")
# args.add_argument("--mr", "-mxrows", default= 10, required=False, help=" max number of rows to be red by data streamer, to make running code faster")




def main(arg):
    # NOTE: eache of these could require a lot of time to complete the job
    # step 1: extract job entities
    print(" Extracting entities ... ")
    #
    extractEntity(arg[1], arg[2], arg[3], int(arg[4]))
    print(" Extracting entities Finished ")

    print(" Calculating similarity matrix ... ")
    # step 2: calculate_similarity
    measure_similarity(arg[1], int(arg[4]) )
    print(" Calculating similarity matrix Finished ")


if __name__ == "__main__":

    # main(args.parse_args())
    main(sys.argv)

print("Finished ... .. .")
