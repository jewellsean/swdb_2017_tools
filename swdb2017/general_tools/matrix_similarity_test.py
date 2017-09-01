import numpy as np 
from scipy import signal
from skimage import measure
from sklearn import cluster

def remove_small_clusters(mtx, labels, remove_size = 20):
	'''
		Remove small connected components 

		input: 

		mtx - n * n binary matrix 
		labels - labels for each connected component
		remove_size - all connected components with fewer units than remove_size are removed

		output: 

		cleaned matrix with all small connected components removed

	'''
	for lbl in np.unique(labels):
		if np.sum(lbl == labels) < remove_size:
			mtx[(labels == lbl)] = 0
	return(mtx)

def standardize(mtx): 
	'''
		Standard Z-transform based on *whole* matrix 

		input: 

		mtx - n * n matrix 

		output 

		z-transformed matrix (zero mean and unit variance)

	'''
	return((mtx - np.mean(mtx)) / np.std(mtx))


def smooth(mtx, kernel_size):
	'''
		Uniform kernel smoothing with mtx 

		input: 

		mtx - n * n matrix 
		kernel_size - matrix of 1s with dimension kernel_size * kernel_size used to smooth mtx

		output: 

		smoothed matrix of same dimension as mtx


	'''
	smooth = np.ones(shape=(kernel_size, kernel_size))
	mtx_smooth = signal.convolve2d(mtx, smooth, boundary='symm', mode='same')
	return(mtx_smooth)

def binarize(mtx, thresh):
	'''
		Binarize matrix based on a threshold 

		input: 

		mtx - n * n matrix 
		thresh - a threshold used to binarize 

		output: 

		an n * n matrix whose (i, j) entry is equal to 1 if mtx[i, j] >= thresh, and 0 otherwise

	'''

	mtx_threshold = mtx.copy()
	mtx_threshold[mtx < thresh] = 0 
	mtx_threshold[mtx >= thresh] = 1
	return(mtx_threshold)

def compute_blocky_metric(mtx, kernel_size = 5, sd_thresh = 1, remove_size = 20):
	'''
		Compute perimeter of all connected components for an input matrix after uniform
		smoothing, Z-standardization, and cleaning for small components. 

		input: 
			
		mtx - n * n matrix 
		kernel_size - size of kernel used in smoothing 
		sd_thresh - threshold to use for binarization
		remove_size - filter on number of cells required in connected components


		output: 

		dict of all intermediary quantities (standardized matrix, binarized matrix, cleaned 
		matrix) and the perimeter of the clean matrix.

	'''
	Z = standardize(mtx.copy())
	mtx_smooth = smooth(Z.copy(), kernel_size)
	Z = standardize(mtx_smooth.copy())
	Z_threshold = binarize(Z.copy(), sd_thresh)
	labels = measure.label(Z_threshold)
	Z_clean = remove_small_clusters(Z_threshold.copy(), labels, remove_size)
	blockyness = measure.perimeter(Z_clean, neighbourhood=4)
	
	## prepare output dict
	out = {}
	out['mtx_standardize'] = Z
	out['mtx_binarized'] = Z_threshold
	out['mtx_clean'] = Z_clean
	out['perimeter'] = blockyness

	return(out)

def permute_and_dist(orig_mtx, params):
	'''

	'''
	n = orig_mtx.shape[0]
	permuted_ind = np.random.permutation(n)
	X_permuted = orig_mtx[permuted_ind, ]
	X_permuted = X_permuted[:, permuted_ind]
	return(compute_blocky_metric(X_permuted, params['kernel_size'], 
		params['sd_thresh'], params['remove_size']))

def calculate_sampling_dist(mtx, params):
	'''
		Calculate the sampling distribution of blockyness based on random permutations
		Output original matrix blocky-ness and the maximal blocky-ness based on bicluster algo

		input: 
		
		mtx - n * n matrix 
		params - dict of parameters, eg {'num_samples':100, 'kernel_size':5, 'sd_thresh':1, 'remove_size':10}
	
		output: 

		dict of the original distance, sampling distances, actual samples, and a p-value
	
	'''
	num_samples = params['num_samples']
	distances = np.zeros(num_samples)
	samples = []
	orig_out = compute_blocky_metric(mtx, params['kernel_size'], 
		params['sd_thresh'], params['remove_size'])
	orig_dist = orig_out['perimeter']

	## calculate dist for every permutation
	for sample_i in range(num_samples):
		permuted_sample = permute_and_dist(mtx, params)
		distances[sample_i] = permuted_sample['perimeter']
		samples.append(permuted_sample['mtx_clean'])

	out = {}
	out['orig_dist'] = orig_dist
	out['sample_distances'] = distances
	out['samples'] = samples
	out['p_value'] =  1 - ((np.sum(distances < orig_dist)) * 1.0) / params['num_samples']

	## max blocky
	cl = cluster.bicluster.SpectralCoclustering(n_clusters = 2, random_state = 0)


	return(out)


