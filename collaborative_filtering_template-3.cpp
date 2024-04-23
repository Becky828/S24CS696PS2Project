#include <iostream>
#include <fstream>
#include <istream>
#include <random>
#include <set>
#include <map>
#include <vector>
#include <sstream>
#include <random>

//#include <cmath>

using namespace std;

//function used to randomly popiualate U and V
//provided vy Dr. Vu
bool toss_coin(double p) {
	// Return true with probability p, false with probability 1-p
	static std::default_random_engine engine(std::random_device{}());
	std::bernoulli_distribution distribution(p);
	return distribution(engine);
}

//function used to generate a uniform random number
//provided by Dr. Vu
double generate_uniform_random_number() {
	// Static to maintain state across function calls and only initialized once
	static std::default_random_engine engine(std::random_device{}());
	std::uniform_real_distribution<double> distribution(0.0, 1);

	// Generate and return the random number
	return distribution(engine);
}

//function used to find the dot product of two vectors
//provided by Dr. Vu
double dot_product(std::vector<double>& v1, std::vector<double>& v2) {
	double result = 0;
	for (int i = 0; i < v1.size(); i++) {
		result += v1[i] * v2[i];
	}
	return result;
}

//function used to find the mean absolute error
//inner workings provided by Dr. Vu
void mae_finder(std::map<std::pair<int, int>, double> test_set, std::vector<std::vector<double>> U, std::vector<std::vector<double>> V)
{
	// calculate the mean absolute error
	double mae = 0;
	double mae_random = 0; // mean absolute error if we were to guess 3 for every rating

	for (auto& x : test_set) {
		int i = x.first.first;
		int j = x.first.second;
		double r = x.second;
		double prediction = dot_product(U[i], V[j]);
		if (prediction > 5) {
			prediction = 5;
		}
		else if (prediction < 1) {
			prediction = 1;
		}
		mae += abs(dot_product(U[i], V[j]) - r);
		mae_random += abs(3 - r);
	}

	mae = static_cast<double>(mae / test_set.size());
	mae_random = static_cast<double>(mae_random / test_set.size());
	std::cout << "Mean Absolute Error: " << mae << std::endl;
	std::cout << "Mean Absolute Error Random Guess: " << mae_random << std::endl;
}



//Custom Functions

//function which finds the transpose of V
std::vector<std::vector<double>>  v_transposer(int K, std::vector<std::vector<double>> V, std::set<int> movies, int n) {

	//initializes the V transposed
	std::vector<std::vector<double>> V_transposed(V[0].size(), std::vector<double>(n, 0));
	for (int j : movies) {
		for (int k = 0; k < K; k++) {
			V_transposed[k][j] = V[j][k];
		}
	}

	return V_transposed;
}


//Collaborative Filtering Stochastic Gradient Descent


//function which performs collaborative filtering stochastic gradient descent
void cf_stochastic_gradient_descent_finder(std::map<std::pair<int, int>, double> test_set, int n_iterations, double eta, double lambda, double decay, std::set<int> users, std::set<int>  movies, std::map<std::pair<int, int>,
	double> ratings, double U_dot_V_transposed, std::map<int, std::set<int>> users_movies, std::map<int, std::set<int>> movies_users, int m, int n, int K, std::vector<std::vector<double>> U, std::vector<std::vector<double>> V) {


	//The following code is based on the collaborative filtering mini-batch gradient descent algorithm as inferred from slide 35 and 38 of the recommendation systems notes
	//slide 23 of the stochastic gradient descent notes


	// initializes mt19937 
	// object for finding random numbers 
	std::mt19937 mt(time(nullptr));

	//performs 0 to n_iterations of collaboarative filtering stochastic gradient descent incrementing by 1
	for (int t = 0; t < n_iterations; t++) {
		eta = eta * decay; // decay the learning rate over time

		//randomly iterates to a rating
		auto it = ratings.begin();
		int random = mt() % ratings.size();
		std::advance(it, random);

		//stores the key of the randomly selected rating
		auto random_key = it->first;

		//stores the randomly selected user
		int i = random_key.first;
		//int current_user = i;

		//stores the randomly selected movie
		int j = random_key.second;

		//iterates through the set of users by an increment of 1. This provides the index required for iterating through the rows of U.
		for (int a : users) {

			//finds the rating difference by subtracting the current rating from the dot product of U and V transposed
			double rating_difference = dot_product(U[a], V[j]) - it->second;

			//iterates through the columns of U by an increment of 1
			for (int k = 0; k < K; k++) {

				//performs the full gradient descent for U
				U[a][k] = U[a][k] - eta *
					((rating_difference * V[j][k]) + (2 * lambda * U[a][k]));
			}
		}

		//iterates through the set of movies by an increment of 1. This provides the index required for iterating through the rows of V.
		for (int b : movies) {

			//finds the rating difference by subtracting the current rating from the dot product of U and V transposed
			double rating_difference = dot_product(U[i], V[b]) - it->second;

			//iterates through the columns of V by an increment of 1
			for (int k = 0; k < K; k++) {

				//performs the full gradient descent for V
				V[b][k] = V[b][k] - eta * ((rating_difference * U[i][k]) + (2 * lambda * V[b][k]));
			}
		}

		//prints the current iteration
		std::cout << "Finished collaborative filtering stochastic gradient descent iteration " << t << endl;

		//finds the current mean absolute error
		mae_finder(test_set, U, V);
	}

	//prints that the collaborative filtering stochastic gradient descent is finished
	std::cout << "Finished Collaborative Filtering Stochastic Gradient Descent" << std::endl;
}

//Collaborative Filtering Batch Gradient Descent

//function which performs collaborative filtering batch gradient descent
void cf_batch_gradient_descent_finder(int n_iterations, std::map<std::pair<int, int>, double> test_set, double eta, double lambda, double decay, std::set<int> users, std::set<int>  movies, std::map<std::pair<int, int>,
	double> ratings, double U_dot_V_transposed, std::map<int, std::set<int>> users_movies, std::map<int, std::set<int>> movies_users, int m, int n, int K, std::vector<std::vector<double>> U, std::vector<std::vector<double>> V) {

	//The following code is based on the collaborative filtering batch gradient descent algorithm as inferred from slide 35 of the recommendation systems notes

	//performs 0 to n_iterations of collaborative filtering gradient descent incrementing by 1
	for (int t = 0; t < n_iterations; t++) {
		eta = eta * decay; // decay the learning rate over time

		// implement gradient descent here:
		// you may want to use for (int i : users) and for (int j : movies) 
		// to iterate over all users and movies instead of for (int i = 0; i < m; i++) and for (int j = 0; j < n; j++)
		// to avoid iterating over users and movies that are not in the training set

		//addition
		// you may also want to use the dot_product function to calculate the dot product of U[i] and V[j]
		// and the derived_u_getter and derived_v_getter functions to calculate the sum of the derived U and V values
		// you can also use the lambda, eta, and decay variables


		//iterates through the set of users by an increment of 1. This provides the index required for iterating through the rows of U
		for (int i : users) {

			//initializes found as true
			bool found = true;

			//initializes the full gradient for U. This ensures that the full gradient for U is set to 0 for each user. The full gradient for U is composed of the base gradient for U and the partial derivative 
			// of the regularization term in respect to U
			std::vector<std::vector<double>>cf_batch_gradient_base_and_norm_U(m, std::vector<double>(K, 0));

			//ensures that the code does not crash if the user does not have any movies
			try {
				users_movies.at(i);
			}

			catch (const out_of_range& e) {

				//sets found to false if the user does not have any movies
				found = false;
			}

			//is entered only if the user has any movie or movies
			if (found) {

				//iterates through all movies in the user's movie set by an increment of 1
				for (int j : users_movies.at(i)) {
					double ratings_difference = dot_product(U[i], V[j]) - ratings.at(std::make_pair(i, j));
					for (int k = 0; k < K; k++) {

						// updates the full gradient for U 
						// by adding the product of the difference between the dot product of U and V transposed and the current rating 
						// and the current element of V 
						// and the product of 2 times lambda and the current element of U (the partial derivative of the regularization term in respect to U[i])
						// to the current element of the full gradient for U	
						cf_batch_gradient_base_and_norm_U[i][k] = cf_batch_gradient_base_and_norm_U[i][k]
							+
							((ratings_difference * V[j][k]) + (2 * lambda * U[i][k]));
					}
				}

				//iterates through all the columns of U by an increment of 1
				for (int k = 0; k < K; k++) {

					//performs the full gradient descent for U
					U[i][k] = U[i][k] - eta * (cf_batch_gradient_base_and_norm_U[i][k]);
				}
			}
		}

		int movies_with_at_most_one_eighth_max_numbers_of_users_in_any_movie_set = 0;
		int max_users_in_any_movie_set = 0;
		int average_number_of_computed_movie_sets = 0;
		for (int j : movies) {
			if (max_users_in_any_movie_set < movies_users[j].size()) {
				max_users_in_any_movie_set = movies_users[j].size();
			}
		}
		//iterates through the set of movies by an increment of 1. This provides the index required for iterating through the rows of V.
		for (int j : movies) {

			//initializes the full gradient for U. This ensures that the full gradient for V is set to 0 for each user. The full gradient for V is composed of the base gradient for V and the partial derivative 
			// of the regularization term in respect to V			
			std::vector<std::vector<double>> cf_batch_gradient_base_and_norm_V(n, std::vector<double>(K, 0));

			//initializes found as true
			bool found = true;

			//ensures that the code does not crash if the movie does not have any users
			try
			{
				movies_users.at(j);
			}
			catch (const out_of_range& e)
			{
				//sets found to false if the movie does not have any users
				found = false;
			}

			//is entered only if the movie has at least one user
			if (found) {

				//	if (movies_users.at(j).size() <= ceil(max_users_in_any_movie_set/1024)) {
						//iterates through all users in the movie's user set by an increment of 1
				for (int i : movies_users.at(j)) {
					++movies_with_at_most_one_eighth_max_numbers_of_users_in_any_movie_set;
					double ratings_difference = dot_product(U[i], V[j]) - ratings.at(std::make_pair(i, j));
					for (int k = 0; k < K; k++) {
						cf_batch_gradient_base_and_norm_V[j][k] = cf_batch_gradient_base_and_norm_V[j][k]
							+
							((ratings_difference * U[i][k]) + (2 * lambda * V[j][k]));
					}
					//	}

						//iterates through all the columns of V by an increment of 1
					for (int k = 0; k < K; k++) {

						// updates the full gradient for V
						// by adding the product of the difference between the dot product of U and V transposed and the current rating 
						// and the current element of U 
						// and the product of 2 times lambda and the current element of V (the partial derivative of the regularization term in respect to V[j])
						// to the current element of the full gradient for V	
						V[j][k] = V[j][k] - eta * (cf_batch_gradient_base_and_norm_V[j][k]);

						//for debugging
						//average_number_of_computed_movie_sets = ceil(movies_with_at_most_one_eighth_max_numbers_of_users_in_any_movie_set / movies.size());
					}
				}
			}
		}

		//average_number_of_computed_movie_sets = ceil(movies_with_at_most_one_eighth_max_numbers_of_users_in_any_movie_set / movies.size());

		//prints the current iteration
		std::cout << "Finished collaborative filtering batch gradient descent iteration " << t << endl;

		//finds the current mean absolute error
		mae_finder(test_set, U, V);
	}

	//prints that the collaborative filtering batch gradient descent is finished
	std::cout << "Finished Collaborative Filtering Batch Gradient Descent" << std::endl;
}


//Collaborative Filtering Mini-Batch Gradient Descent

//function which performs collborative filtering  mini-batch gradient descent
void cf_mini_batch_gradient_descent_finder(int batch_size, std::map<std::pair<int, int>, double> test_set, int n_iterations, double eta, double lambda, double decay, std::set<int> users, std::set<int>  movies, std::map<std::pair<int, int>,
	double> ratings, double U_dot_V_transposed, std::map<int, std::set<int>> users_movies, std::map<int, std::set<int>> movies_users, int m, int n, int K, std::vector<std::vector<double>> U, std::vector<std::vector<double>> V) {


	//The following code is based on the collaborative filtering mini-batch gradient descent algorithm as inferred from slide 35 and 38 of the recommendation systems notes
	//slide 23 of the stochastic gradient descent notes

	// initializes mt19937 
	// object for finding random numbers  
	std::mt19937 mt(time(nullptr));


	//performs 0 to n_iterations of mini-batch gradient descent incrementing by 1
	for (int t = 0; t < n_iterations; t++) {
		eta = eta * decay; // decay the learning rate over time

		//initializes batch as a subset of ratings keys
		std::map<std::pair<int, int>, double> ratings_batch;

		//iterates through the batch size by an increment of 1 to find a subset of ratings
		for (int a = 0; a < batch_size; a++) {

			//randomly iterates to a rating
			auto it = ratings.begin();
			int random = mt() % ratings.size();
			std::advance(it, random);

			//for debugging
			//random = rand() % ratings.size();

			//stores the key of the randomly selected rating			
			ratings_batch.insert(*it);
		}

		//iterates through the set of users by an increment of 1. This provides the index required for iterating through the rows of U
		for (int a : users) {

			//initializes the full gradient for U. This ensures that the full gradient for U is set to 0 for each user. The full gradient for U is composed of the base gradient for U and the partial derivative 
			// of the regularization term in respect to U
			std::vector<std::vector<double>>cf_mini_batch_gradient_base_and_norm_U(m, std::vector<double>(K, 0));

			//iterates through all the columns of U by an increment of 1
			//for (int k = 0; k < K; k++) {

			//performs the summation of the base gradient for a subset of samples relating to U
			for (auto it : ratings_batch) {

				//stores i and j
				int i = it.first.first;
				int j = it.first.second;

				//finds the current rating difference by subtracting the current rating from the dot product of U and V transposed
				double rating_difference = dot_product(U[a], V[j]) - it.second;

				//iterates through the columns of U by an increment of 1
				for (int k = 0; k < K; k++) {

					// updates the base gradient for U 
					// by adding the product of the difference between the dot product of U and V transposed and the current rating 
					// and the current element of V 
					// and the product of 2 times lambda and the current element of U (the partial derivative of the regularization term in respect to U[i])
					// to the current element of the full gradient for U						
					cf_mini_batch_gradient_base_and_norm_U[a][k] = cf_mini_batch_gradient_base_and_norm_U[a][k]
						+
						((rating_difference * V[j][k]) + (2 * lambda * U[a][k]));
				}

				for (int k = 0; k < K; k++) {

					//performs the base gradient descent for U
					//the base gradient for U is divided by the batch size to find the average base gradient for U
					U[a][k] = U[a][k] - eta * (cf_mini_batch_gradient_base_and_norm_U[a][k] / batch_size);
				}
			}
		}

		//iterates through the set of movies by an increment of 1. This provides the index required for iterating through the rows of V.
		for (int a : movies) {

			//initializes the full gradient for V. This ensures that the full gradient for V is set to 0 for each user. The full gradient for V is composed of the base gradient for V and the partial derivative 
			// of the regularization term in respect to V
			std::vector<std::vector<double>>cf_mini_batch_gradient_base_and_norm_V(n, std::vector<double>(K, 0));

			//iterates through all the columns of V by an increment of 1
			//for (int k = 0; k < K; k++) {

				//performs the summation of the base gradient for a subset of samples relating to V
			for (auto it : ratings_batch) {

				//stores i and j
				int i = it.first.first;
				int j = it.first.second;

				//finds the current rating difference
				double rating_difference = dot_product(U[i], V[a]) - it.second;

				//iterates through the columns of V by an increment of 1
				for (int k = 0; k < K; k++) {

					// updates the base gradient for V
					// by adding the product of the difference between the dot product of U and V transposed and the current rating 
					// and the current element of U 
					// and the product of 2 times lambda and the current element of V (the partial derivative of the regularization term in respect to V[j])
					// to the current element of the full gradient for V							
					cf_mini_batch_gradient_base_and_norm_V[a][k] = cf_mini_batch_gradient_base_and_norm_V[a][k]
						+
						((rating_difference * U[i][k]) + (2 * lambda * V[a][k]));
				}

				for (int k = 0; k < K; k++) {

					//performs the base gradient descent for V 
					//the base gradient for V is divided by the batch size to find the average base gradient for V
					V[a][k] = V[a][k] - eta * (cf_mini_batch_gradient_base_and_norm_V[a][k] / batch_size);
				}
			}
		}

		//prints the current iteration
		std::cout << "Finished collaborative filtering mini-batch iteration " << t << endl;

		//finds the current mean absolute error
		mae_finder(test_set, U, V);
	}

	//prints that the gradient descent is finished
	std::cout << "Finished Collaborative Filtering Mini-Batch Gradient Descent" << std::endl;
}

//function which performs various methods of collaborative filtering gradient descent
int main() {

	//for quick debugging
	//std::ifstream file("very_abridged_Dataset.csv");

	//for first part of p2
	std::ifstream file("Dataset.csv");
	// std::ifstream file("Movie_Id_Titles.csv");    

	//for second part of p2
	//std::ifstream file("ratings.csv");

	std::string line;

	//initializes the ratings
	std::map<std::pair<int, int>, double> ratings;

	//initializes the test set
	std::map<std::pair<int, int>, double> test_set;

	//initializes the users_movies
	std::map<int, std::set<int>> users_movies;

	//initializes the movies_users
	std::map<int, std::set<int>> movies_users;


	//initializes the users
	std::set<int> users;

	//initializes the movies
	std::set<int> movies;

	//Full Dataset
	int K = 15; // number of latent dimensions	
	int m = 2000; // upper bound for number of users	
	int m_bonus = 270600; // upper bound for number of users
	int n = 2000; // upper bound number of movies	
	int n_bonus = 6800; // upper bound number of movies
	int batch_size = 0; // batch size

	//Abirdged Dataset
	//int K = 15; // number of latent dimensions
	//int m = 500; // upper bound for number of users
	//int n = 500; // upper bound number of movies

	double test_set_size = 0.1; // percentage of the data will be used for testing
	double test_set_size_bonus = 0.1; // percentage of the data will be used for testing
	double one_percent = 0.01; // percentage of the data will be used for testing
	double five_percent = 0.05; // percentage of the data will be used for testing
	double ten_percent = 0.1; // percentage of the data will be used for testing
	double twenty_percent = 0.2; // percentage of the data will be used for first level fine tuning the hyperparameters
	double thirty_percent = 0.3; // percentage of the data will be used for second level fine tuning the hyperparameters
	double forty_percent = 0.4; // percentage of the data will be used for third level fine tuning the hyperparameters
	double lambda = 1e-3; // regularization parameter
	double lambda_copy = lambda;
	double lambda_10_times_up = lambda * 10;
	double lambda_10_times_down = lambda / 10;
	double eta = 1e-4; // learning rate
	double eta_copy = eta;
	double eta_10_times_up = eta * 10;
	double eta_10_times_down = eta / 10;
	double decay = 0.9; // decay rate
	int n_iterations = 35; // number of iterations for the gradient descent
	int n_iterations_copy = n_iterations;
	int n_iterations_double = 2 * n_iterations;
	double U_dot_V_transposed = 0;
	double V_dot_U = 0;

	//int epochs = 4;

//	std::vector<std::vector<std::vector<double>>> updated_U_V;


	// read the userids, movieids, and ratings from the file for the main part
	if (file.is_open()) {
		std::getline(file, line); // skip the first line

		while (std::getline(file, line)) {

			std::istringstream iss(line);
			std::string token;
			// read user, movie, and rating
			std::getline(iss, token, ',');
			int user = std::stol(token);
			std::getline(iss, token, ',');
			int movie = std::stol(token);
			std::getline(iss, token, ',');
			double rating = std::stod(token);

			if (toss_coin(1 - test_set_size)) {
				// if the coin toss is true, add the rating to the training set
				ratings[std::make_pair(user, movie)] = rating;
				//double current_rating_a = ratings[std::make_pair(user, movie)];
				users_movies[user].insert(movie); // add movie to user's list of movies
				movies_users[movie].insert(user); // add user to movie's list of users
			}
			else {
				// if the coin toss is false, add the rating to the test set
				test_set[std::make_pair(user, movie)] = rating;
			}

			// keep track of users and movies that have been added
			// the Ids might be larger than the number of users and movies
			users.insert(user);
			movies.insert(movie);
		}

		file.close();
	}
	else {
		std::cout << "Unable to open file" << std::endl;
	}

	std::cout << "Finish Reading File" << std::endl;



	//	batch_size = ratings.size() * 0.010;

	// initialize U and V for the collaborative filtering
	std::vector<std::vector<double>> U(m, std::vector<double>(K, 0));
	std::vector<std::vector<double>> V(n, std::vector<double>(K, 0));

	// initialize copiesa of U and V for U and V reset
	std::vector<std::vector<double>> copy_U(m, std::vector<double>(K, 0));
	std::vector<std::vector<double>> copy_V(n, std::vector<double>(K, 0));


	batch_size = ratings.size() * 0.010;


	// initialize U and V with random values
	for (int i : users) {
		for (int k = 0; k < K; k++) {
			U[i][k] = generate_uniform_random_number();
		}
	}

	for (int j : movies) {
		for (int k = 0; k < K; k++) {
			V[j][k] = generate_uniform_random_number();
		}
	}

	//initialize the copy of U and V
	copy_U = U;
	copy_V = V;

	n_iterations = n_iterations_copy;
	eta = eta_copy;
	lambda = lambda_copy;



	// initialize u and v with random values
	for (int i : users) {
		for (int k = 0; k < k; k++) {
			U[i][k] = generate_uniform_random_number();
		}
	}

	for (int j : movies) {
		for (int k = 0; k < k; k++) {
			V[j][k] = generate_uniform_random_number();
		}
	}


	//initialize the copy of u and v
	copy_U = U;
	copy_V = V;


	//p2a 
	// Collaborative Filtering Batch Gradient Descent

	//resetting U and V
	U = copy_U;
	V = copy_V;

	//setting number of iterations appears to reduce the MAE by around 0.01
	n_iterations = n_iterations_copy;

	//resets the eta to the original value
	eta = eta_copy;


	// 1 of 5
	// collaborative filtering batch gradient descent found with given hyperparameters
	std::cout << "\n" << "\n" << "Collaborative Filetering Batch Gradient Descent:" << std::endl;
	std::cout << "\n" << "1 of 5:" << std::endl;
	std::cout << "Given Hyperparameters" << std::endl;
	cf_batch_gradient_descent_finder(n_iterations, test_set, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, users_movies, movies_users, m, n, K, U, V);
	std::cout << "1 of 5." << std::endl;

	//resetting U and V
	U = copy_U;
	V = copy_V;

	//doubling the number of iterations appears to reduce the MAE by around 0.01
	n_iterations = n_iterations_double;

	//resets the eta to the original value
	eta = eta_copy;

	// 2 of 5
	//collaborative filtering batch gradient descent found with the doubled number of iterations
	std::cout << "\n" << "2 of 5:" << std::endl;
	std::cout << "Doubled Number of Iterations" << std::endl;
	cf_batch_gradient_descent_finder(n_iterations, test_set, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, users_movies, movies_users, m, n, K, U, V);
	std::cout << "2 of 5." << std::endl;


	//resetting U and V
	U = copy_U;
	V = copy_V;

	//muliplying the eta by 10 appears to reduce the MAE by around  0.15
	eta = eta_10_times_up;

	// 3 of 5
	//collborative batch gradient descent found with the doubled number of iterations and eta times 10
	std::cout << "\n" << "3 of 5:" << std::endl;
	std::cout << "Doubled Number of Iterations, eta times 10, and unchanged lambda" << std::endl;

	//finds uodated U and V
	cf_batch_gradient_descent_finder(n_iterations, test_set, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, users_movies, movies_users, m, n, K, U, V);
	std::cout << "3 of 5." << std::endl;


	//resetting U and V
	U = copy_U;
	V = copy_V;

	//resets the eta to 10 times the original value
	eta = eta_10_times_up;

	//muliplying the lambda by 10 appears to slightly increase the MAE
	lambda = lambda_10_times_up;

	//4 of 5
	//collaborative filtering batch gradient descent found with the doubled number of iterations, eta times 10, and lambda times 10
	std::cout << "\n" << "4 of 5:" << std::endl;
	std::cout << "Doubled Number of Iterations, eta times 10, and lambda times 10" << std::endl;
	std::cout << "This provided the lowest found MAE." << std::endl;

	//gets updated V and U
	cf_batch_gradient_descent_finder(n_iterations, test_set, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, users_movies, movies_users, m, n, K, U, V);
	std::cout << "4 of 5." << std::endl;


	//resetting U and V
	U = copy_U;
	V = copy_V;

	//resets the eta to 10 times the original value
	eta = eta_10_times_up;

	//dividing the lambda by 10 appears to slightly increase the MAE
	lambda = lambda_10_times_down;

	//5 of 5
	//collaborative filtering batch gradient descent found with the doubled number of iterations, eta times 10, and lambda divided by 10
	std::cout << "\n" << "5 of 5:" << std::endl;
	std::cout << "Doubled Number of Iterations, eta times 10, and lambda divided by 10" << std::endl;

	//gets updated V and U
	cf_batch_gradient_descent_finder(n_iterations, test_set, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, users_movies, movies_users, m, n, K, U, V);
	std::cout << "5 of 5." << std::endl;


	//p2b-i
	//Collaborative Filtering Stochastic Gradient Descent

	//eta = eta_copy;
	//n_iterations = 100000;
	//n_iterations = 60;
	//n_iterations = 70;
	//n_iterations = m;
	////eta = eta * (1000 + 57.4);
	////eta = eta * 900;
	//eta = eta * 9000;

	//lambda = lambda / 90000;

	//resetting U and V
	U = copy_U;
	V = copy_V;

	//setting the eta to 1000 times the original value
	eta = 1000 * eta_copy;
	//lambda = lambda / 90;
	lambda = lambda_10_times_up;
	n_iterations = 6 * n_iterations_copy;
	//n_iterations = m;
	//epochs = 100;
	//lambda = lambda_copy;

	std::cout << "\n" << "\n" << "Collaborative Filtering Stochastic Gradient Descent:" << std::endl;
	std::cout << "\nRuns five times with the same hyperparameters to demonstrate the effects of randomness produced by mt19937" << std::endl;
	std::cout << "\n" << "1 of 5:" << std::endl;
	cf_stochastic_gradient_descent_finder(test_set, n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, users_movies, movies_users, m, n, K, U, V);
	std::cout << "1 of 5." << std::endl;


	//resetting U and V
	U = copy_U;
	V = copy_V;

	//setting the eta to 1000 times the original value
	eta = 1000 * eta_copy;
	//lambda = lambda / 90;
	lambda = lambda_10_times_up;
	//n_iterations = 5 * n_iterations;
	//n_iterations = m;
	//epochs = 100;
	//lambda = lambda_copy;

	std::cout << "\n" << "2 of 5:" << std::endl;
	cf_stochastic_gradient_descent_finder(test_set, n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, users_movies, movies_users, m, n, K, U, V);
	std::cout << "2 of 5." << std::endl;

	//resetting U and V
	U = copy_U;
	V = copy_V;

	//setting the eta to 1000 times the original value
	eta = 1000 * eta_copy;
	//lambda = lambda / 90;
	lambda = lambda_10_times_up;
	//n_iterations = 5 * n_iterations;
	//n_iterations = m;
	//epochs = 100;
	//lambda = lambda_copy;

	std::cout << "\n" << "3 of 5:" << std::endl;
	cf_stochastic_gradient_descent_finder(test_set, n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, users_movies, movies_users, m, n, K, U, V);
	std::cout << "3 of 5." << std::endl;


	//resetting U and V
	U = copy_U;
	V = copy_V;

	//setting the eta to 1000 times the original value
	eta = 1000 * eta_copy;
	//lambda = lambda / 90;
	lambda = lambda_10_times_up;
	//n_iterations = 5 * n_iterations;
	//n_iterations = m;
	//epochs = 100;
	//lambda = lambda_copy;

	std::cout << "\n" << "4 of 5:" << std::endl;
	cf_stochastic_gradient_descent_finder(test_set, n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, users_movies, movies_users, m, n, K, U, V);
	std::cout << "4 of 5." << std::endl;



	//resetting U and V
	U = copy_U;
	V = copy_V;

	//setting the eta to 1000 times the original value
	eta = 1000 * eta_copy;
	//lambda = lambda / 90;
	lambda = lambda_10_times_up;
	//n_iterations = 5 * n_iterations;
	//n_iterations = m;
	//epochs = 100;
	//lambda = lambda_copy;

	std::cout << "\n" << "5 of 5:" << std::endl;
	cf_stochastic_gradient_descent_finder(test_set, n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, users_movies, movies_users, m, n, K, U, V);
	std::cout << "5 of 5." << std::endl;


	//p2-ii
	//Collaborative Filtering Stochastic Gradient Descent

	//resetting U and V
	U = copy_U;
	V = copy_V;

	//eta = eta_copy;
	//lambda = lambda_copy;
	//eta = eta_10_times_up*100;
	eta = eta_copy * 1000;
	batch_size = ratings.size() * 0.01;
	std::cout << "\n" << "\n" << "Collaborative Filtering Mini-Batch Gradient Descent:" << std::endl;
	std::cout << "Using the Same Hyperparameters as Collaborative Filtering Stochastic Gradient Descent \nto Analyze Similarity of the Two Methods" << std::endl;
	std::cout << "\nRuns two times with the same hyperparameters to demonstrate the effects of randomness produced by mt19937" << std::endl;
	std::cout << "\n" << "1 of 2:" << std::endl;
	cf_mini_batch_gradient_descent_finder(batch_size, test_set, n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, users_movies, movies_users, m, n, K, U, V);
	std::cout << "1 of 2." << std::endl;

	//resetting U and V
	U = copy_U;
	V = copy_V;

	//eta = eta_copy;
	//lambda = lambda_copy;
	//eta = eta_10_times_up*100;
	eta = eta_copy * 1000;

	std::cout << "\n" << "2 of 2:" << std::endl;
	cf_mini_batch_gradient_descent_finder(batch_size, test_set, n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, users_movies, movies_users, m, n, K, U, V);
	std::cout << "2 of 2." << std::endl;

	std::cout << "\n" << "\n" << "End of PS2 p2 Main Part:" << std::endl;






	
	
		//p2 bonus
		//Collaborative Filtering Batch Gradient Descent on 0.25 of the Bonus Data
	
		std::cout << "\n" << "\n" << "Bonus" << std::endl;
	
		m = 7000; // upper bound for number of users
		n = 300000; // upper bound number of movies
	
		//reinitializes the U and V for the bonus
		U.assign(m, std::vector<double>(K, 0));
		V.assign(n, std::vector<double>(K, 0));
	
		//empties the ratings
		ratings.clear();
	
		//empties the test set
		test_set.clear();
	
		//empties the users_movies
		users_movies.clear();
	
		//empties the movies_users
		movies_users.clear();
	
		//empties the users
		users.clear();
	
		//empties the movies
		movies.clear();
	
		//resets n_iterations
		n_iterations = n_iterations_double;
	
		//resets the eta to 10 times the original value
		eta = eta_10_times_up;
	
		//muliplying the lambda by 10 appears to slightly increase the MAE
		lambda = lambda_10_times_up;
	
	
		//first level hyperparameter fine tuning
		std::ifstream file_bonus("ratings.csv");
	
		if (file_bonus.is_open()) {
			std::getline(file_bonus, line); // skip the first line
	
			while (std::getline(file_bonus, line)) {
	
				std::istringstream iss(line);
				std::string token;
				// read user, movie, and rating
				std::getline(iss, token, ',');
				int user = std::stol(token);
				std::getline(iss, token, ',');
				int movie = std::stol(token);
				std::getline(iss, token, ',');
				double rating = std::stod(token);
	
				if (toss_coin(twenty_percent)) {
					if (toss_coin(1 - test_set_size)) {
						// if the coin toss is true, add the rating to the training set
						ratings[std::make_pair(user, movie)] = rating;
						//double current_rating_a = ratings[std::make_pair(user, movie)];
						users_movies[user].insert(movie); // add movie to user's list of movies
						movies_users[movie].insert(user); // add user to movie's list of users
					}
					else {
						// if the coin toss is false, add the rating to the test set
						test_set[std::make_pair(user, movie)] = rating;
					}
	
					// keep track of users and movies that have been added
					// the Ids might be larger than the number of users and movies
					users.insert(user);
					movies.insert(movie);
				}
			}
			//file_bonus.close();
	
			//flushes the file_bonus
			file_bonus.clear();
		}
		else {
			std::cout << "Unable to open file" << std::endl;
		}
	
		std::cout << "Finish Bonus File Read 1 of 4" << std::endl;
		
		//	batch_size = ratings.size() * 0.010;
	
		// initialize U and V with random values
		for (int i : users) {
			for (int k = 0; k < K; k++) {
				U[i][k] = generate_uniform_random_number();
			}
		}
	
		for (int j : movies) {
			for (int k = 0; k < K; k++) {
				V[j][k] = generate_uniform_random_number();
			}
		}
	
		//1 of 4 
		//collaborative filtering batch gradient descent first level of hyperparameter fine tuning for batch gradient descent
		std::cout << "\n" << "1 of 4:" << std::endl;
		std::cout << "Doubled Number of Iterations, eta times 10, and lambda times 10" << std::endl;
	
		//gets updated V and U
		cf_batch_gradient_descent_finder(n_iterations, test_set, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, users_movies, movies_users, m, n, K, U, V);
		std::cout << "1 of 4." << std::endl;
	
	
		
		//second level hyperparameter fine tuning
		
		//returns to the beginning of the file
		file_bonus.seekg(0, file_bonus.beg);
	
		//reinitializes the U and V for the bonus
		U.assign(m, std::vector<double>(K, 0));
		V.assign(n, std::vector<double>(K, 0));
	
		//empties the ratings
		ratings.clear();
	
		//empties the test set
		test_set.clear();
	
		//empties the users_movies
		users_movies.clear();
	
		//empties the movies_users
		movies_users.clear();
	
		//empties the users
		users.clear();
	
		//empties the movies
		movies.clear();
	
		//resets n_iterations
		n_iterations = n_iterations_double;
	
		//resets the eta to 10 times the original value
		eta = eta_10_times_up;
	
		//muliplying the lambda by 10 appears to slightly increase the MAE
		lambda = lambda_10_times_up;
	
	
		if (file_bonus.is_open()) {
			std::getline(file_bonus, line); // skip the first line
	
			while (std::getline(file_bonus, line)) {
	
				std::istringstream iss(line);
				std::string token;
				// read user, movie, and rating
				std::getline(iss, token, ',');
				int user = std::stol(token);
				std::getline(iss, token, ',');
				int movie = std::stol(token);
				std::getline(iss, token, ',');
				double rating = std::stod(token);
	
				//uses thirty percent of the dataset
				if (toss_coin(thirty_percent)) {
					if (toss_coin(1 - test_set_size)) {
						// if the coin toss is true, add the rating to the training set
						ratings[std::make_pair(user, movie)] = rating;
						//double current_rating_a = ratings[std::make_pair(user, movie)];
						users_movies[user].insert(movie); // add movie to user's list of movies
						movies_users[movie].insert(user); // add user to movie's list of users
					}
					else {
						// if the coin toss is false, add the rating to the test set
						test_set[std::make_pair(user, movie)] = rating;
					}
	
					// keep track of users and movies that have been added
					// the Ids might be larger than the number of users and movies
					users.insert(user);
					movies.insert(movie);
				}
			}
	
			//flushes the file_bonus
			file_bonus.clear();
		}
		else {
			std::cout << "Unable to open file" << std::endl;
		}
	
		std::cout << "Finish Bonus File Read 2 of 4" << std::endl;
		
	
		// initialize U and V with random values
		for (int i : users) {
			for (int k = 0; k < K; k++) {
				U[i][k] = generate_uniform_random_number();
			}
		}
	
		for (int j : movies) {
			for (int k = 0; k < K; k++) {
				V[j][k] = generate_uniform_random_number();
			}
		}
	
	
		//2 of 4 
		//collaborative filtering batch gradient descent second level of hyperparameter fine tuning for batch gradient descent
		std::cout << "\n" << "2 of 4:" << std::endl;
		std::cout << "Doubled Number of Iterations, eta times 10, and lambda times 10" << std::endl;
	
		//gets updated V and U
		cf_batch_gradient_descent_finder(n_iterations, test_set, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, users_movies, movies_users, m, n, K, U, V);
		std::cout << "2 of 4." << std::endl;
	
		
	
		//third level hyperparameter fine tuning
		
		//returns to the beginning of the file
		file_bonus.seekg(0, file_bonus.beg);
	
	
		//reinitializes the U and V for the bonus
		U.assign(m, std::vector<double>(K, 0));
		V.assign(n, std::vector<double>(K, 0));
	
		//empties the ratings
		ratings.clear();
	
		//empties the test set
		test_set.clear();
	
		//empties the users_movies
		users_movies.clear();
	
		//empties the movies_users
		movies_users.clear();
	
		//empties the users
		users.clear();
	
		//empties the movies
		movies.clear();
	
		//resets n_iterations
		n_iterations = n_iterations_double;
	
		//resets the eta to 10 times the original value
		eta = eta_10_times_up;
	
		//muliplying the lambda by 10 appears to slightly increase the MAE
		lambda = lambda_10_times_up;
	
		if (file_bonus.is_open()) {
			std::getline(file_bonus, line); // skip the first line
	
			while (std::getline(file_bonus, line)) {
	
				std::istringstream iss(line);
				std::string token;
				// read user, movie, and rating
				std::getline(iss, token, ',');
				int user = std::stol(token);
				std::getline(iss, token, ',');
				int movie = std::stol(token);
				std::getline(iss, token, ',');
				double rating = std::stod(token);
	
				//uses forty percent of the dataset
				if (toss_coin(forty_percent)) {
					if (toss_coin(1 - test_set_size)) {
						// if the coin toss is true, add the rating to the training set
						ratings[std::make_pair(user, movie)] = rating;
						//double current_rating_a = ratings[std::make_pair(user, movie)];
						users_movies[user].insert(movie); // add movie to user's list of movies
						movies_users[movie].insert(user); // add user to movie's list of users
					}
					else {
						// if the coin toss is false, add the rating to the test set
						test_set[std::make_pair(user, movie)] = rating;
					}
	
					// keep track of users and movies that have been added
					// the Ids might be larger than the number of users and movies
					users.insert(user);
					movies.insert(movie);
				}
			}
	
			//flushes the file_bonus
			file_bonus.clear();
		}
		else {
			std::cout << "Unable to open file" << std::endl;
		}
	
		std::cout << "Finish Bonus File Read 2 of 4" << std::endl;
	
	
		// initialize U and V with random values
		for (int i : users) {
			for (int k = 0; k < K; k++) {
				U[i][k] = generate_uniform_random_number();
			}
		}
	
		for (int j : movies) {
			for (int k = 0; k < K; k++) {
				V[j][k] = generate_uniform_random_number();
			}
		}
	
	
		//3 of 4 
		//collaborative filtering batch gradient descent third level of hyperparameter fine tuning for batch gradient descent
		std::cout << "\n" << "3 of 4:" << std::endl;
		std::cout << "Doubled Number of Iterations, eta times 10, and lambda times 10" << std::endl;
	
		//gets updated V and U
		cf_batch_gradient_descent_finder(n_iterations, test_set, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, users_movies, movies_users, m, n, K, U, V);
		std::cout << "3 of 4." << std::endl;
	
	
		//fourth level hyperparameter fine tuning
		//returns to the beginning of the file
		file_bonus.seekg(0, file_bonus.beg);
	
	
		//reinitializes the U and V for the bonus
		U.assign(m, std::vector<double>(K, 0));
		V.assign(n, std::vector<double>(K, 0));
	
		//empties the ratings
		ratings.clear();
	
		//empties the test set
		test_set.clear();
	
		//empties the users_movies
		users_movies.clear();
	
		//empties the movies_users
		movies_users.clear();
	
		//empties the users
		users.clear();
	
		//empties the movies
		movies.clear();
	
		//resets n_iterations
		n_iterations = n_iterations_double;
	
		//resets the eta to 10 times the original value
		eta = eta_10_times_up;
	
		//muliplying the lambda by 10 appears to slightly increase the MAE
		lambda = lambda_10_times_up;
	
		//second level hyperparameter fine tuning
		if (file_bonus.is_open()) {
			std::getline(file_bonus, line); // skip the first line
	
			while (std::getline(file_bonus, line)) {
	
				std::istringstream iss(line);
				std::string token;
				// read user, movie, and rating
				std::getline(iss, token, ',');
				int user = std::stol(token);
				std::getline(iss, token, ',');
				int movie = std::stol(token);
				std::getline(iss, token, ',');
				double rating = std::stod(token);
	
				//uses full dataset
					if (toss_coin(1 - test_set_size)) {
						// if the coin toss is true, add the rating to the training set
						ratings[std::make_pair(user, movie)] = rating;
						//double current_rating_a = ratings[std::make_pair(user, movie)];
						users_movies[user].insert(movie); // add movie to user's list of movies
						movies_users[movie].insert(user); // add user to movie's list of users
					}
					else {
						// if the coin toss is false, add the rating to the test set
						test_set[std::make_pair(user, movie)] = rating;
					}
	
					// keep track of users and movies that have been added
					// the Ids might be larger than the number of users and movies
					users.insert(user);
					movies.insert(movie);
				
			}
			//file_bonus.close();
	
			file_bonus.clear();
		}
		else {
			std::cout << "Unable to open file" << std::endl;
		}
	
		std::cout << "Finish Bonus File Read 4 of 4" << std::endl;
	
		//close the file
		file_bonus.close();
	
	
		// initialize U and V with random values
		for (int i : users) {
			for (int k = 0; k < K; k++) {
				U[i][k] = generate_uniform_random_number();
			}
		}
	
		for (int j : movies) {
			for (int k = 0; k < K; k++) {
				V[j][k] = generate_uniform_random_number();
			}
		}
	
	
		//4 of 4 
		//collaborative filtering batch gradient descent first level of hyperparameter fine tuning for batch gradient descent
		std::cout << "\n" << "4 of 4:" << std::endl;
		std::cout << "Doubled Number of Iterations, eta times 10, and lambda times 10" << std::endl;
	
		//gets updated V and U
		cf_batch_gradient_descent_finder(n_iterations, test_set, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, users_movies, movies_users, m, n, K, U, V);
		std::cout << "4 of 4." << std::endl;




	//returns 0 to indicate that the program has ran without errors
	return 0;

}