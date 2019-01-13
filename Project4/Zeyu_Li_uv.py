import sys
import numpy as np
import math

if __name__ == "__main__":


	with open(sys.argv[1]) as f:
	 	lines = f.readlines()

	# with open('mat.dat') as f:
	#	lines = f.readlines()

	n = int(sys.argv[2])
	m = int(sys.argv[3])
	f = int(sys.argv[4])
	it = int(sys.argv[5])
	# n = 5
	# m = 5
	# f = 2
	# it = 10

	# initialize target matrix
	mat = np.zeros(shape=(n, m))
	for i in lines:
		tmp = i.split(',')
		mat[int(tmp[0]) - 1][int(tmp[1]) - 1] = int(tmp[2])

	# print mat
	# initialize U, V
	U = np.ones(shape=(n, f))
	V = np.ones(shape=(f, m))

	# UV decomp
	for z in range(it):
		# training U
		for r in range(n): # r is row index of U
			for s in range(f): # s is col index of U
				tmpsum = 0
				tmpd = 0
				for j in range(m):
					tmpsum2 = 0
					if mat[r][j] > 0:
						for k in range(f):
							if k != s:
								tmpsum2 += U[r][k] * V[k][j]
						tmpsum += V[s][j] * (mat[r][j] - tmpsum2)
						tmpd += V[s][j] * V[s][j]
				U[r][s] = float(tmpsum) / tmpd
				# print 'U(' + str(r+1) + ',' + str(s+1) + ') = ' + str(U[r][s])

		for s in range(m):
			for r in range(f):
				tmpsum = 0
				tmpd = 0
				for i in range(n):
					tmpsum2 = 0
					if mat[i][s] > 0:
						for k in range(f):
							if k != r:
								tmpsum2 += U[i][k] * V[k][s]
						tmpsum += U[i][r] * (mat[i][s] - tmpsum2)
						tmpd += U[i][r] * U[i][r]
				V[r][s] = float(tmpsum) / tmpd
				# print 'V(' + str(r+1) + ',' + str(s+1) + ') = ' + str(V[r][s])
		
		# print cost
		# print V
		approx = np.dot(U, V)
		# print approx
		error = 0
		cnt = 0
		for i in range(n):
			for j in range(m):
				if mat[i][j] > 0:
					error += (mat[i][j] - approx[i][j]) * (mat[i][j] - approx[i][j])
					cnt += 1
		print cnt
		error /= float(cnt)
		error = math.sqrt(error)
		print "%.4f" %error




























