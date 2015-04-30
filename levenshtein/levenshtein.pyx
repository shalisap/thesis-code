"""
Levenshtein distance between 2 series
@author: Julian Applebaum
"""

def uniformMatrix(r, c, v=0):
	"""
	Create a c x r matrix filled with the value v
	@param r: number of rows
	@param c: number of columns
	@param v: the value to fill with
	@return: the matrix
	"""
	matrix = []
	for i in xrange(0, r):
		matrix.append([v for i in xrange(0,c)])
	return matrix

cdef int levRec(seq1, seq2, int i, int j, memo):
	"""
	Recursive helper for levDistance.
	@param seq1: a sequence of numbers
	@param seq2: a sequence of numbers
	@param i: the index to cut seq1 at - start at |seq1|
	@param j: the index to cut seq2 at - start at |seq2|
	@param memo: 2d array for memoizing computations
	@return: The Levenshtein distance between seq1 and seq2
	"""
	if min(i, j) == 0:
		if i > j:
			return sum(seq1[0:i]) + i
		else:
			return sum(seq2[0:j]) + j
	elif memo[i-1][j-1] != -1:
		return memo[i-1][j-1]
	else:
		lev_del = levRec(seq1, seq2, i-1, j, memo) + 1 + seq1[i-1]
		lev_ins = levRec(seq1, seq2, i, j-1, memo) + 1 + seq2[j-1]
		lev_sub = levRec(seq1, seq2, i-1, j-1, memo) + abs(seq1[i-1] - seq2[j-1])
		dist = min(lev_del, lev_ins, lev_sub)
		memo[i-1][j-1] = dist
		return dist

def levDistance(pair):
	"""
	Compute the Levenshtein distance between two time series (seq1, seq2).
	Insertions and deletions cost 1 + value inserted/deleted. Substitutions
	cost abs(val_seq1 - val_seq2).
	@param pair: a tuple (seq1, seq2)
	@return: the Levenshtein distance between seq1 and seq2
	"""
	seq1, seq2 = pair
	cdef int i_init
	cdef int j_init
	i_init = len(seq1)
	j_init = len(seq2)
	memo = uniformMatrix(i_init, j_init, -1)
	return levRec(seq1, seq2, i_init, j_init, memo)
