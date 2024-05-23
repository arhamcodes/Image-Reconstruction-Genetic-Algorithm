import numpy as np
import cv2

#def fitness_fun(solution, solution_idx):
#   fitness = np.sum(np.abs(target_chromosome-solution))
#   fitness = np.sum(target_chromosome) - fitness
#   return fitness

def Fitness(chromosome, target):
    fitness = np.mean(np.abs(chromosome-target))
    return fitness

def Crossover(chrom1, chrom2):
    point = np.random.randint(0, len(chrom1))
    new_chrom1 = np.concatenate((chrom1[:point], chrom2[point:]))
    new_chrom2 = np.concatenate((chrom2[:point], chrom1[point:]))
    return new_chrom1, new_chrom2

def Mutation(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            chromosome[i] = np.random.randint(0, 256)
    return chromosome

x_data = np.array(cv2.imread("./batman2.png"))
chromosome = x_data.flatten()
population = np.random.randint(0, 256, (100, len(chromosome)))
fitness = {}
mutation_rate = 0.01    

for j in range(300):
    for i in range(len(population)):
        fitness[i] = Fitness(population[i], chromosome)
    best_chromosomes = sorted(fitness, key=fitness.get)[:2]
    new_chrom1, new_chrom2 = Crossover(population[best_chromosomes[0]], population[best_chromosomes[1]])
    new_chrom1 = Mutation(new_chrom1, mutation_rate)
    new_chrom2 = Mutation(new_chrom2, mutation_rate)
    population[sorted(fitness, key=fitness.get)[np.random.randint(0, 100)]] = new_chrom1
    population[sorted(fitness, key=fitness.get)[np.random.randint(0, 100)]] = new_chrom2

new_image = population[sorted(fitness, key=fitness.get)[0]].reshape(x_data.shape)
cv2.imwrite("new_image.png", new_image)