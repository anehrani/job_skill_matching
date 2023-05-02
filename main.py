
"""
    main function to run the entity extraction and similarity score generation

"""

from src.MatchSkills import extractEntity
from src.MeasureSimilarity import measure_similarity



# NOTE: eache of these could require a lot of time to complete the job
# step 1: extract job entities
extractEntity()

# step 2: calculate_similarity
measure_similarity()





print("Finished ... .. .")
