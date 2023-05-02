# job_skill_matching

To ran use provided bashscript "create_environment_and_run_code.sh" FOR LINUX SYSTEMS
this script creates a python3.8 virtual env and install the requirements and run the code.
To sea if the code is running on smalle chunk of data, we can limit the number of data to be red from the data files
max_rows: max number of rows to be red. 
If you want to run the code for all data set max_rows = -1
To run bash script:

create_environment_and_run_code.sh path/to/job/description/file path/to/skills/file path/to/tech/skills/file max_rows
example:  
bash create_environment_and_run_code.sh data/jobpostings.csv data/Skills.slsx data/Technology_Skills.xlsx 10



This will generate results and save them under results folder.

Considerations and Possible Implementations:
* This code could be very slow, so parallelization is necessary to run faster.
* Data streaming is suitable for reading big data as it just loads a row every time

 
 In this repo I am using pretrained language models to map text into an algebraic space and then use existing algorithms to 
 find the possible similarities between description.


#Future improvements:
This code loads all of the data  to the memory to calculate the similarity between descriptions.
This method wont work for huge data, to solve that there are clustering techniques; where we can iteratively create clusters and keep only small part of the data on the memory.

To extract entities from job description, current code will work with minor modification as im only streaming small amount of data at the same time 
without need to the rest of the data.
#Accuracy:
I am currently using pretrained naive codes. In order to increase the accuracy we can fine tune existing models on our dataset for both cases. 

NOTE: My suggestion is not to use python language for manipulating Big data. Although, python is very good for testing the models, it is very slow in manipulating big data. This is not because algorithm running slower on python as already algorithms are implemented in c.
I suggest using Rust or C++ instead.   
  

