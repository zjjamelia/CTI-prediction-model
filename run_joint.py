import numpy as np
import scipy.io as sio
import time

from joint import joint

max_iter = 20
restart_prob = 0.50

drug_nets = ['Sim_mat_drug_matrix', 'Sim_mat_drug_protein', 'Sim_mat_drug_structure']
protein_nets = ['Sim_mat_protein_matrix','Sim_mat_protein_drug','Resized_Sim_mat_protein_protein']

start_time = time.time()
print("这里",f'{drug_nets}.txt')
X = joint(drug_nets, restart_prob, max_iter)
end_time = time.time()
print(f"Time for drug networks: {end_time - start_time} seconds")

start_time = time.time()
Y = joint(protein_nets, restart_prob, max_iter)
end_time = time.time()
print(f"Time for protein networks: {end_time - start_time} seconds")

np.savetxt('../feature1/drug_vector.txt', X, delimiter='\t')
np.savetxt('../feature1/protein_vector.txt', Y, delimiter='\t')