import numpy as np

def permute(orig_mtx):
	'''

	'''
	n = orig_mtx.shape[0]
	X = np.array(orig_mtx)
	permuted_ind = np.random.permutation(n)
	X_permuted = X[permuted_ind, ]
	X_permuted = X_permuted[:, permuted_ind]
	return(X_permuted)

def matrix_similarity(model_mtx, mtx):
	return(np.corrcoef(model_mtx.flatten(), mtx.flatten())[1][0])
	

def calculate_sampling_dist(model_mtx, emp_mtx, num_samples):
	'''

		input: 
		
		model_mtx - n * n model matrix 
		exp_mtx - n * n RSA matrix
		n_samples - number of random pertubations 
	
		output: 

		dict of the original s, sampling distances, actual samples, and a p-value
	
	'''
	similarities = np.zeros(num_samples)
	samples = []
	orig_sim = matrix_similarity(model_mtx, emp_mtx)

	## calculate simularity for every permutation
	for sample_i in range(num_samples):
		permuted_sample = permute(emp_mtx)
		similarities[sample_i] = matrix_similarity(model_mtx, permuted_sample)
		samples.append(permuted_sample)

	out = {}
	out['orig_sim'] = orig_sim
	out['samples'] = samples
	out['similarities'] = similarities
	out['p_value'] =  1 - ((np.sum(similarities < orig_sim)) * 1.0) / num_samples

	return(out)
