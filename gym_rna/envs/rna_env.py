# Environment: A mimic of the nature
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from utils import *
from matplotlib import pyplot as plt

# bases = ['A', 'U', 'G', 'C']
MAP = [('A', 'U'), ('C', 'G'), ('G', 'U')]
MAP = [('A', 'U'), ('C', 'G')]
MAP += [(v, u) for (u, v) in MAP]
MAP = list(set(MAP))

VALUES = {('A', 'U'): 1, ('C', 'G'): 2, ('G', 'U'): 0}
for (u, v) in VALUES.keys():
	VALUES[(v, u)] = VALUES[(u, v)]

class RNA():
	def __init__(self, sequence, right_pairs=MAP, min_pairing_gap=3):
		"""
		@inputs:
			sequence: primary structure of the strand
		"""
		self.right_pairs = right_pairs
		self.sequence = sequence
		self.onehot_seq = seq_to_onehot(sequence, bases)
		self.mapping_matrix = create_mapping_matrix(self.right_pairs, bases)
		self.N = len(sequence)
		self.pairing_list = range(len(sequence))
		self.pairing_matrix = self.create_pairing_matrix()
		self.min_pairing_gap = min_pairing_gap
		self.fe = 0

	def __str__(self):
		"""
		Override the str function just for printing the RNA as a string
		"""
		return ''.join(self.sequence)

	def __len__(self):
		"""
		Override the len function to get the length of the sequence
		"""
		return len(self.sequence)

	def create_pairing_matrix(self, pairing=None):
		"""
		Create the pairing matrix using the secondary structure pairing vector
		"""
		if not pairing:
			pairing = self.pairing_list
		mat = np.zeros((self.N+4, self.N+4))
		for i in range(self.N):
			j = pairing[i]
			mat[i][j] = 1
		mat[-4:, :-4] = self.onehot_seq.T
		mat[:-4, -4:] = self.onehot_seq
		mat[-4:, -4:] = self.mapping_matrix
		return mat

	def is_paired(self, i):
		"""
		Return True if base i paired to any other base
		"""
		return self.pairing_list[i] != i

	def get_pair(self, i):
		"""
		Get the paired index
		-1 if the base was free
		"""
		return self.pairing_list[i]

	def is_right_pairing(self, i, j):
		"""
		Return True for right pairing like C-G
		and False for wrong pairing like A-G
		** w.r.t the right_pairs
		"""
		if (self.sequence[i], self.sequence[j]) not in self.right_pairs:
			return False
		if abs(i - j) < self.min_pairing_gap+1:
			return False
		return self.pairing_list[i] == i and self.pairing_list[j] == j
		

	def do_pairing(self, i, j, mode='secondary'):
		"""
		Pain i and j basis to each other 
		Update secondary list and matrix
		"""
		################
		# DANGEROUS LINE
		#
		#
		#
		if i == j:
			return self.do_unpairing(i)
		#
		#
		#
		################
		elif self.is_right_pairing(i, j):
			if mode == 'secondary':
				if not self.is_cross(i, j):
					# print self.pairing_list, i, j
					# print self.pairing_matrix
					self.pairing_list[i] = j
					self.pairing_list[j] = i
					self.pairing_matrix[i, i] = 0
					self.pairing_matrix[i, j] = 1
					self.pairing_matrix[j, j] = 0
					self.pairing_matrix[j, i] = 1
					return 2
				else:
					return 0
			
		else:
			return 0
			# print "Wrong pairing!"

	def do_unpairing(self, i):
		"""
		Unpair the i base with its corresponding base
		If it was not paired, do nothing!!!
		Then, update the structure
		"""
		j = self.pairing_list[i]
		if i == j:
			return 0
		self.pairing_list[i] = i
		self.pairing_list[j] = j
		self.pairing_matrix[i, j] = 0
		self.pairing_matrix[i, i] = 1
		self.pairing_matrix[j, i] = 0
		self.pairing_matrix[j, j] = 1
		return -2

	def is_cross(self, i, j):
		"""
		Return True if adding the i-j pair cross anything
		"""
		# It means there exists k: i<k<j s.t. is connected to somewhere out of (i, j)
		if i > j:
			i, j = j, i
		for k in range(i, j):
			if self.pairing_list[k] < i or self.pairing_list[k] > j:
				return True
		return False
		# return (np.sum(self.pairing_matrix[i:j, :i]) + np.sum(self.pairing_matrix[i:j, j:])) > 0

	def to_dot_parentheses(self, pairing=None):
		"""
		Convert the arc diagram to the dot pranthesis structure
		******* NOTE: For now, only works for secondary structure
		"""
		if not pairing:
			pairing = self.pairing_list
		dot = ''
		for (i, si) in zip(range(len(pairing)), pairing):
			if si == i:
				dot += '.'
			elif si < i:
				dot += ')'
			else:
				dot += '('
		return dot

	def free_energy(self):
		# return self.to_dot_parentheses().count('(')
		fe = free_energy(self.sequence, self.to_dot_parentheses())
		return fe
		new_fe = free_energy(self.sequence, self.to_dot_parentheses())
		reward = self.fe - new_fe
		self.fe = new_fe
		return reward

	def mfe(self, seq):
		fc = RNA.fold_compound(seq)
		ss, e = fc.mfe()
		return e




class RNAEnv(gym.Env):
	def __init__(self, sequence='', pairing_values=VALUES, log=False):
		self.pairing_values = pairing_values
		self.reset(sequence)
		# self.safety_zone = safety_zone

	def reset(self, sequence):
		self.rna = RNA(sequence)
		self.N = len(sequence)
		return self.rna.pairing_matrix

	def step(self, a, with_reward=False):
		"""
		Pair/unpair the i and j together
		return:
			s_prime (next state)
			reward
			done
		"""
		
		i = a / self.N
		j = a % self.N
	
		j_prime = self.rna.get_pair(i)
		# print i, j, j_prime
		if j_prime == i: # i is free and can connect to j
			r = self.rna.do_pairing(i, j)
			# return self.rna.pairing_matrix, r, 0
			
			# return self.rna.pairing_matrix, self.rna.free_energy(), 0
		elif j_prime != j: #i is already connected to a different j
			# Do nothing
			r = 0
			# return self.rna.pairing_matrix, 0, 0
		else: # They are connected, so unpair them
			r = self.rna.do_unpairing(i)
			# return self.rna.pairing_matrix, -2, 0
		# return self.rna.pairing_matrix, r, 0 
		if with_reward:
			r = self.rna.free_energy()
		else:
			r = 0
		
		return self.rna.pairing_matrix, r, 0
		return np.triu(self.rna.pairing_matrix), R, 0
			

	def generate_random_structure(self, n_pairs, max_iter=200):
		self.reset(self.rna.sequence)
		pairs = 0
		i = 0
		while pairs<n_pairs and i<max_iter:
			i += 1
			pairs += self.step(np.random.randint(self.N**2))[1]/2
		return self.rna.pairing_matrix


def k_diag(mat, k):
	idx = np.diag_indices(mat.shape[0])
	if k == 0:
		return idx
	elif k>0:
		return (idx[0][:-k], idx[1][k:])
	else:
		return (idx[0][-k:], idx[1][:k])


if __name__ == '__main__':
	seq = 'CCCCCCUUUUUUGGGGGG'
	# seq = 'CCCCCCAAAAAAGGGGGGUUUUUU'
	seq = generate_random_seq(50, .5)
	seq = 'AAAAUUAUAGCAGUCAAUGCGUCGCAUCACUCCGAAUUUCCCGCACCGUG'
	print seq
	N = len(seq)
	env = Environment(seq)
	# structure = env.generate_random_structure(2)

	m = env.rna.pairing_matrix
	# plt.matshow(m, cmap=plt.cm.Blues)
	r = np.matmul(m[:-4, -4:], m[-4:, -4:])
	r = np.matmul(r, m[-4:, :-4])

	for i in range(-3, 4, 1):
		r[k_diag(r, i)] = 0

	for i in range(1, N-1, 1):
		for j in range(1, N-1, 1):
			r[i, j] = r[i, j] * max(r[i+1, j-1], r[i-1, j+1])

	np.fill_diagonal(r, 1)
	# print r 
	# r = np.maximum(r, np.eye(N))
	plt.matshow(r, cmap=plt.cm.Blues)
	plt.axis('off')
	plt.show()


	# rules = np.zeros((N, N))
	# for i in range(N):
	# 	print m[i, -4:] #* m[-4:, -4:]

