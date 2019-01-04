"""
Mehdi Saman Booy
For the ready-to-use function purposes
"""
import numpy as np
from os import system
from random import choice

bases = ['A', 'U', 'G', 'C']

def free_energy(sequence, structure, max_fe=100.):
	system('echo "%s\n%s" > tmp_input.in' % (sequence, structure))
	system('export NUPACKHOME=/usr/local/nupack; $NUPACKHOME/build/bin/energy tmp_input > tmp_energy')
	# system('/usr/local/nupack/build/bin/energy tmp_input')
	with open('tmp_energy', 'r') as energy_f:
		energy = energy_f.readlines()[-1]
	# try:
	e = float(energy[:-2])
	return min(e, max_fe)
	# except:
		# return False


def seq_to_onehot(sequence, bases):
	mapping = [bases.index(q) for q in sequence]
	n = len(sequence)
	onehot = np.zeros((n, len(bases)))
	onehot[np.arange(n), mapping] = 1
	return onehot


def create_mapping_matrix(right_pairs, bases):
	n = len(bases)
	mapping = np.zeros((n, n))
	for (i, j) in right_pairs:
		iidx,  jidx = bases.index(i), bases.index(j)
		mapping[iidx, jidx] = 1
		mapping[jidx, iidx] = 1
	return mapping


def generate_random_seq(n, gc_content=.5):
	# Let's do some crazy things :-D
	g_prob = gc_content/2.
	a_prob = .5 - g_prob
	return ''.join(np.random.choice(bases, n, p=[a_prob, a_prob, g_prob, g_prob]))


if  __name__ == '__main__':
	seq = 'CCCGGGAAAUUU'
	# print seq_to_onehot(seq, bases)
	print generate_random_seq(32, .5)



