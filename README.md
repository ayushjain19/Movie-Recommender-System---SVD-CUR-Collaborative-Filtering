# Movie-Recommender-System---SVD-CUR-Collaborative-Filtering

This directory contains code for implementation of Collaborative filtering, SVD, CUR.
The complement code is implemented in Python and requires python3 for execution

The packages used in the execution of the program are:
* numpy
* math
* operator
* time

*The code is divided into 9 functions as follows:*
1. **collaborative_filtering_func**: Function that carries out collaborative filtering
algorithm
1. **find_similarity**: It gives the similarity between two vectors
1. **svd_func**: Function to initiate the SVD algorithm
1. **get_new_VT**: This function shrinks the size of VT when 90% energy is being
maintained in SVD algorithm
1. **predict**: This function is called in svd_func function. It carries out the task of
predicting the testing data
1. **cur_func**: This function initiates the CUR algorithm
1. **select_random_rows**: This function is used to select random rows from a
given matrix where each row can have a probability of being selected.
1. **find_U_and_rmse**: This function finds the U matrix and also calculates
RMSE, Precision and Spearman Rank Correalation
1. **get_top_k_movies**: This function is used to get the top most rated movies. Its
used in finding the Precision


**Results obtained:**

*Collaborative without baseline approach:*
* RMSE: 0.0768
* Precision on top 50: 0.76
* Spearman Rank Correlation: 0.999704867798
* Time taken: 36.071 secs

*Collaborative with baseline approach:*
* RMSE: 0.0802659721
* Precision on top 50: 0.76
* Spearman Rank Correlation: 0.999677846314
* Time taken: 34.13582 secs

*SVD:*
* RMSE: 0.011385705026
* Precision on top 50: 0.12
* Spearman Rank Correlation: 0.999999992222
* Time taken: 8.547 secs

*SVD with 90% retained energy:*
* RMSE: 0.0113993709
* Precision on top 50: 0.12
* Spearman Rank Correlation: 0.999999992203
* Time taken: 3.1909 secs
