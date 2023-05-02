# job_skill_matching

To ran use provided bashscript "create_environment_and_run_code.sh" FOR LINUX SYSTEMS
this script creates a python3.8 virtual env and install the requirements and run the code.

To run bash script:

create_environment_and_run_code.sh path/to/job/description/file path/to/skills/file path/to/tech/skills/file
example:  
bash create_environment_and_run_code.sh data/jobpostings.csv data/Skills.slsx data/Technology_Skills.xlsx

This will generate results and save them under results folder.

Considerations and Possible Implementations:
* This code could be very slow, so parallelization is necessary to run faster.
 
