import numpy as np
from numpy import linalg as la
from numpy.linalg import svd
import math
import operator
import time


# Function to get top k movies
def get_top_k_movies(temp, k):
	movie_index_rating = []
	top_k_movies_for_temp = []
	avg_rating_of_movie = np.zeros(len(temp[0]))
	for j in range(len(temp[0])):
		number_of_users_rated = 0
		num = 0
		for i in range(len(temp)):
			if(temp[i][j] != 0):
				number_of_users_rated += 1
				num += temp[i][j]
		if(number_of_users_rated > 0):
			avg_rating_of_movie[j] = float(num) / number_of_users_rated
			movie_index_rating.append([j, avg_rating_of_movie[j]])

	sorted_movie_index_rating = sorted(movie_index_rating, key = operator.itemgetter(1), reverse = True)

	for i, index in zip(range(k), range(len(sorted_movie_index_rating))):
		top_k_movies_for_temp.append(sorted_movie_index_rating[i][0])

	return top_k_movies_for_temp


# Similarity function
def find_similarity(X, Y):
	numerator = 0.0
	sum_of_square_of_components_of_X = 0.0
	sum_of_square_of_components_of_Y = 0.0
	
	for i in range(len(X)):
		numerator += X[i] * Y[i]
		sum_of_square_of_components_of_X += X[i] ** 2
		sum_of_square_of_components_of_Y += Y[i] ** 2

	denomenator = math.sqrt(sum_of_square_of_components_of_X) * math.sqrt(sum_of_square_of_components_of_Y)
	if(denomenator == 0):
		return 0
	else:
		return float(numerator) / denomenator


# Collaborative filtering function
def collaborative_filtering_func(AT, BT, no_of_neighbors, movies_rated_by_user, to_be_predicted, temp, k, top_k_movies_for_B, baseline_approach):

	print("In collaborative filtering function!")
	movie_offset = np.zeros(len(AT))
	mean_movie_rating = 0.0
	total_rating = 0.0
	number_of_ratings = 0

	# Finding mean movie rating throughout matrix
	for i in range(len(AT)):
		for j in range(len(AT[i])):
			if(AT[i][j] != 0):
				total_rating += AT[i][j]
				number_of_ratings += 1
	mean_movie_rating = float(total_rating) / number_of_ratings

	rating_deviation_of_user = np.zeros(len(AT[0]))
	rating_deviation_of_movie = np.zeros(len(AT))


	# Rating deviation of each user
	# Given by: Average rating of user - mean movie rating
	for j in range(len(AT[0])):
		num = 0.0
		number_of_movies_rated = 0
		for i in range(len(AT)):
			if(AT[i][j] != 0):
				num += AT[i][j]
				number_of_movies_rated += 1
		if(number_of_movies_rated > 0):
			rating_deviation_of_user[j] = (float(num) / number_of_movies_rated) - mean_movie_rating


	# Rating deviation of each movie
	# Given by: Average rating of movie - mean movie rating
	for i in range(len(AT)):
		num = 0.0
		number_of_users_rated = 0
		for j in range(len(AT[i])):
			if(AT[i][j] != 0):
				num += AT[i][j]
				number_of_users_rated += 1
		if(number_of_users_rated > 0):
			rating_deviation_of_movie[j] = (float(num) / number_of_users_rated) - mean_movie_rating


	# Normalizing rows of AT here
	for i in range(len(AT)):
		num = 0.0
		no_of_users_rated_current_movie = 0
		for j in range(len(AT[i])):
			if (AT[i][j] != 0):
				num += AT[i][j]
				no_of_users_rated_current_movie += 1
		if(no_of_users_rated_current_movie > 0):
			movie_offset[i] = float(num / float(no_of_users_rated_current_movie))
		for j in range(len(AT[i])):
			if(AT[i][j] != 0):
				AT[i][j] = AT[i][j] - movie_offset[i]


	number_of_predictions = 0
	sum_of_squared_error = 0.0
	count = 0
	
	# Predicting the ratings here
	for data in to_be_predicted:
		# data is of the form [movie, user]
		if(count == int(0.25 * len(to_be_predicted))):
			print("25% data predicted!")
		elif(count == int(0.5 * len(to_be_predicted))):
			print("50% data predicted!")
		elif(count == int(0.75 * len(to_be_predicted))):
			print("75% data predicted!")
	
		count += 1
		sim = []
		for movie in movies_rated_by_user[data[1]]:
			sim.append([movie, find_similarity(AT[data[0]], AT[movie])])
		sorted_sim = sorted(sim, key = operator.itemgetter(1), reverse = True)
		numerator = 0
		denomenator = 0
		for l, i in zip(range(no_of_neighbors), range(len(sorted_sim))):
			if(baseline_approach == True):
				numerator += sorted_sim[l][1] * (BT[sorted_sim[l][0]][data[1]] - (mean_movie_rating + rating_deviation_of_user[data[1]] + rating_deviation_of_movie[sorted_sim[l][0]]))
			else:
				numerator += sorted_sim[l][1] * (BT[sorted_sim[l][0]][data[1]] - movie_offset[sorted_sim[l][0]])
			denomenator += sorted_sim[l][1]
		if(denomenator > 0):
			if(baseline_approach == True):
				rating = mean_movie_rating + rating_deviation_of_user[data[1]] + rating_deviation_of_movie[data[0]] + (numerator / float(denomenator))
			else:
				rating = (numerator / float(denomenator)) + movie_offset[data[0]]
			sum_of_squared_error += (rating - BT[data[0]][data[1]]) ** 2
			temp[data[1]][data[0]] = rating
			number_of_predictions += 1

	
	# Root mean square
	rmse = math.sqrt(sum_of_squared_error) / number_of_predictions
	n = len(to_be_predicted)

	# Spearman Coorelation
	spearman_rank_correlation = 1 - ((6 * sum_of_squared_error) / (n * (n*n - 1)))
	
	count = 0
	
	top_k_movies_for_temp = get_top_k_movies(temp, k)
	for movie in top_k_movies_for_B:
		if(movie in top_k_movies_for_temp):
			count += 1
	print("count: " + str(count))
	print("k: " + str(k))
	precision_on_top_k = float(count) / k


	# Printing the results
	if(baseline_approach):
		print("RMSE for Collaborative filtering with baseline approach: " + str(rmse))
		print("Spearman Rank Correlation for Collaborative filtering with baseline approach: " + str(spearman_rank_correlation))
		print("Precision on top k for Collaborative filtering with baseline approach: " + str(precision_on_top_k))
	else:
		print("RMSE for Collaborative filtering without baseline approach: " + str(rmse))
		print("Spearman Rank Correlation for Collaborative filtering without baseline approach: " + str(spearman_rank_correlation))
		print("Precision on top k for Collaborative filtering without baseline approach: " + str(precision_on_top_k))	

	print("Exiting collaborative filtering function!")
	return


# Predict function for SVD
def predict(A, B, VT, user_offset, temp, k, top_k_movies_for_B):
	V = VT.T
	number_of_predictions = 0
	squared_error_sum = 0
	for i in range(len(A)):
		qV = np.dot(A[i], V)
		rating_for_q = np.dot(qV, VT)
		rating_for_q = rating_for_q + user_offset[i]

		for j in range(len(A[i])):
			if(B[i][j] != 0 and A[i][j] + user_offset[i] != B[i][j]):
				number_of_predictions += 1
				squared_error_sum += (rating_for_q[j] - B[i][j]) ** 2
				temp[i][j] = rating_for_q[j]
	frobenius_norm = math.sqrt(squared_error_sum)
	print("No. of predictions: " + str(number_of_predictions))

	# Root mean square
	rmse = float(frobenius_norm / float(number_of_predictions))
	
	count = 0
	top_k_movies_for_temp = get_top_k_movies(temp, k)
	for movie in top_k_movies_for_B:
		if(movie in top_k_movies_for_temp):
			count += 1
	print("count: " + str(count))
	print("k: " + str(k))
	precision_on_top_k = float(count) / k

	return number_of_predictions, precision_on_top_k, squared_error_sum, rmse



def get_new_VT(VT, eigen_values):

	temp = []
	sum_of_squared_eigenvalues = 0.0
	for i in range(len(eigen_values)):
		temp.append([i, eigen_values[i]])
		sum_of_squared_eigenvalues += eigen_values[i] ** 2
	sorted_eigenvalues = sorted(temp, key = operator.itemgetter(1), reverse = True)
	allowed_loss_of_energy = 0.1 * sum_of_squared_eigenvalues
	
	sum = 0
	for i in range(len(eigen_values)):
		if(sum + eigen_values[-i-1] ** 2 < allowed_loss_of_energy):
			sum += eigen_values[-i-1] ** 2
		else:
			number_of_rows_to_be_retained_in_VT = len(eigen_values) - i
			break

	new_VT = np.zeros((number_of_rows_to_be_retained_in_VT, len(VT[0])))

	for i in range(number_of_rows_to_be_retained_in_VT):
		for j in range(len(VT[i])):
			new_VT[i][j] = VT[temp[i][0]][j]

	return new_VT


# SVD function
def svd_func(A, B, user_offset, temp, k, top_k_movies_for_B):
	print("In SVD function!")
	complex_count = 0
	A_transpose = A.T

	start_time = time.time()
	U, eigen_values, VT = svd(A, full_matrices = False)
	sigma = np.zeros((len(A_transpose), len(A_transpose)))

	for i in range(len(eigen_values)):
		sigma[i][i] = math.sqrt(eigen_values[i])

	temp_time = start_time - time.time()
	n, precision_on_top_k, squared_error_sum, rmse = predict(A, B, VT, user_offset, temp, k, top_k_movies_for_B)
	print("Time taken by SVD: " + str(time.time() - start_time))
	print("RMSE for SVD: " + str(rmse))	

	# Finding Spearman Rank Correlation for SVD
	spearman_rank_correlation = 1 - ((6 * squared_error_sum) / (n * (n*n - 1)))
	print("Spearman Rank Correlation for SVD: " + str(spearman_rank_correlation))
	print("Precision on top k for SVD: " + str(precision_on_top_k))

	start_time = time.time()
	VT = get_new_VT(VT, eigen_values)

	n, precision_on_top_k, squared_error_sum, rmse = predict(A, B, VT, user_offset, temp, k, top_k_movies_for_B)		# Here VT is the new VT after 90% retained energy
	print("Time taken by SVD after 90% retained energy: " + str(time.time() - start_time + temp_time))
	
	print("RMSE for SVD after 90% retained energy: " + str(rmse))

	# Finding Spearman Rank Correlation for SVD after 90% retained energy
	spearman_rank_correlation = 1 - ((6 * squared_error_sum) / (n * (n*n - 1)))
	print("Spearman Rank Correlation for SVD after 90% retained energy: " + str(spearman_rank_correlation))
	print("Precision on top k for SVD after 90% retained energy: " + str(precision_on_top_k))

	print("Exiting SVD function")
	return


# Function to select random rows for CUR
def select_random_rows(B, r, isRepeatationAllowed):
	
	indices = [i for i in range(len(B))]
	square_of_frobenius_norm_of_B = 0
	for i in range(len(B)):
		for j in range(len(B[i])):
			square_of_frobenius_norm_of_B += B[i][j] ** 2

	p = np.zeros(len(B))
	for i in range(len(B)):
		sum_of_squared_values_in_row = 0
		for j in range(len(B[i])):
			sum_of_squared_values_in_row += B[i][j] ** 2
		p[i] = sum_of_squared_values_in_row / float(square_of_frobenius_norm_of_B)

	rows_selected = np.random.choice(indices, r, isRepeatationAllowed, p)

	R = np.zeros((r, len(B[0])))
	for i, row in zip(range(r), rows_selected):
		for j in range(len(B[row])):
			R[i][j] = B[row][j]
			R[i][j] = R[i][j] / float(math.sqrt(r*p[row]))

	return rows_selected, R


def find_U_and_rmse(B, r, row_indices, R, column_indices, C, k, top_k_movies_for_B):
	
	W = np.zeros((r, r))
	for i, row in zip(range(len(row_indices)), row_indices):
		for j, column in zip(range(len(column_indices)), column_indices):
			W[i][j] = B[row][column]

	X, eigen_values, YT = svd(W, full_matrices = False)

	sigma = np.zeros((r, r))
	sigma_plus = np.zeros((r, r))

	for i in range(len(eigen_values)):
		sigma[i][i] = math.sqrt(eigen_values[i])
		if(sigma[i][i] != 0):
			sigma_plus[i][i] = 1 / float(sigma[i][i])

	U = np.dot(np.dot(YT.T, np.dot(sigma_plus, sigma_plus)), X.T)

	# CUR matrix
	cur_matrix = np.dot(np.dot(C, U), R)
	
	count = 0
	top_k_movies_for_cur = get_top_k_movies(cur_matrix, k)
	for movie in top_k_movies_for_B:
		if(movie in top_k_movies_for_cur):
			count += 1
	print("count: " + str(count))
	print("k: " + str(k))
	precision_on_top_k = float(count) / k

	squared_error_sum = 0
	number_of_predictions = 0

	for i in range(len(B)):
		for j in range(len(B[i])):
			if(B[i][j] != 0):
				squared_error_sum += (B[i][j] - cur_matrix[i][j]) ** 2
				number_of_predictions += 1

	print(number_of_predictions)
	frobenius_norm = math.sqrt(squared_error_sum)
	print(frobenius_norm)

	# Root mean square
	rmse = frobenius_norm / float(number_of_predictions)
	return number_of_predictions, precision_on_top_k, squared_error_sum, rmse


# CUR function
def cur_func(B, r, k, top_k_movies_for_B):
	print("In CUR function!")
	start_time = time.time()
	row_indices, temp_matrix = select_random_rows(B, r, True)
	R = temp_matrix
	column_indices, temp_matrix = select_random_rows(B.T, r, True)
	C = temp_matrix.T

	n, precision_on_top_k, squared_error_sum, rmse = find_U_and_rmse(B, r, row_indices, R, column_indices, C, k, top_k_movies_for_B)
	print("RMSE for CUR with rows and columns repeatations: " + str(rmse))

	# Finding Spearman Rank Correlation for CUR with rows and columns repeatations
	spearman_rank_correlation = 1 - ((6 * squared_error_sum) / (n * (n*n - 1)))
	print("Spearman Rank Correlation for CUR with rows and columns repeatations: " + str(spearman_rank_correlation))
	print("Precision on top k for CUR with rows and columns repeatations: " + str(precision_on_top_k))
	print("Time taken for CUR with rows and columns repeatations: " + str(time.time() - start_time))

	start_time = time.time()
	row_indices, temp_matrix = select_random_rows(B, r, False)
	R = temp_matrix
	column_indices, temp_matrix = select_random_rows(B.T, r, False)
	C = temp_matrix.T

	n, precision_on_top_k, squared_error_sum, rmse = find_U_and_rmse(B, r, row_indices, R, column_indices, C, k, top_k_movies_for_B)
	print("RMSE for CUR without rows and columns repeatations: " + str(rmse))

	# Finding Spearman Rank Correlation for CUR without rows and columns repeatations
	spearman_rank_correlation = 1 - ((6 * squared_error_sum) / (n * (n*n - 1)))
	print("Spearman Rank Correlation for CUR without rows and columns repeatations: " + str(spearman_rank_correlation))
	print("Precision on top k for CUR without rows and columns repeatations: " + str(precision_on_top_k))
	print("Time taken for CUR without rows and columns repeatations: " + str(time.time() - start_time))

	print("Exiting CUR function!")
	return




user_ids_index = {}
movie_ids_index = {}
user_count = 0
movie_count = 0
count = 0
max_user_no = 0
max_movie_no = 0
movies_rated_by_user = {}
to_be_predicted = []
k = 50
r = 300

# Reading file for finding max movie id and max user id
with open("u.data", "r") as data_file:
	for line in data_file:
		count += 1
		line_values = line.split("\t")
		a = int(line_values[0])
		b = int(line_values[1])
		if(a > max_user_no):
			max_user_no = a
		if(b > max_movie_no):
			max_movie_no = b

three_fourth_data_length = int(0.75 * count)
counter = 0
count_thousand_data_points = 0
A = np.zeros((max_user_no + 1, max_movie_no + 1))
temper = np.zeros((max_user_no + 1, max_movie_no + 1))
B = np.zeros((max_user_no + 1, max_movie_no + 1))


# Reading file
with open("u.data", "r") as data_file:
	for line in data_file:
		line_values = line.split("\t")
		a = int(line_values[0])
		b = int(line_values[1])
		B[a][b] = float(line_values[2])
		if(counter <= three_fourth_data_length):
			A[a][b] = float(line_values[2])
			temper[a][b] = float(line_values[2])
			counter += 1
			if a not in movies_rated_by_user:
				movies_rated_by_user[a] = [b]
			else:
				movies_rated_by_user[a].append(b)
		elif(count_thousand_data_points < 120):
			to_be_predicted.append([b, a])
			count_thousand_data_points += 1

data_file.close()


# Getting top k rated movies for B
top_k_movies_for_B = get_top_k_movies(B, k)

no_of_neighbors = 5
temp = temper.copy()
start_time = time.time()

# Calling Colloborative function without baseline approach
collaborative_filtering_func(A.T, B.T, no_of_neighbors, movies_rated_by_user, to_be_predicted, temp, k, top_k_movies_for_B, False)
print("Time taken by Collaborative filtering without baseline approach: " + str(time.time() - start_time))
print("")
start_time = time.time()

# Calling Colloborative function with baseline approach
collaborative_filtering_func(A.T, B.T, no_of_neighbors, movies_rated_by_user, to_be_predicted, temp, k, top_k_movies_for_B, True)
print("Time taken by Collaborative filtering with baseline approach: " + str(time.time() - start_time))
print("")
user_offset = np.zeros(max_user_no + 1)

# Normalizing A matrix
for i in range(max_user_no + 1):
	num = 0.0
	no_of_movies_rated_by_current_user = 0
	for j in range(max_movie_no + 1):
		if (A[i][j] != 0):
			num += A[i][j]
			no_of_movies_rated_by_current_user += 1
	if(no_of_movies_rated_by_current_user > 0):
		user_offset[i] = float(num / float(no_of_movies_rated_by_current_user))
	for j in range(max_movie_no + 1):
		if(A[i][j] != 0):
			A[i][j] = A[i][j] - user_offset[i]

temp = temper.copy()

# Calling SVD function
svd_func(A, B, user_offset, temp, k, top_k_movies_for_B)

print("")

# Calling CUR function
cur_func(B, r, k, top_k_movies_for_B)

# Program ends