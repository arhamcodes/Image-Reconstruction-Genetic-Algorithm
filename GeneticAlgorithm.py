import numpy as np
import random

def create_individual(shape, target=None, guided=False):
    if guided and target is not None:
        individual = np.copy(target)
        perturbation = np.random.randint(-50, 50, shape, dtype=np.int32)
        individual = np.clip(individual + perturbation, 0, 255).astype(np.uint8)
    else:
        individual = np.random.randint(0, 256, shape, dtype=np.uint8)
    return individual

def create_population(pop_size, shape, target=None):
    population = []
    guided_proportion = 0.25
    guided_count = int(pop_size * guided_proportion)
    
    for _ in range(guided_count):
        population.append(create_individual(shape, target=target, guided=True))
    
    for _ in range(pop_size - guided_count):
        population.append(create_individual(shape))
    
    return population

def calculate_fitness(individual, target):
    return np.mean(np.abs(individual - target))

def tournament_selection(population, fitnesses, k=3):
    selected = random.choices(list(zip(population, fitnesses)), k=k)
    selected.sort(key=lambda x: x[1]) 
    return selected[0][0]

def crossover(parent1, parent2):
    crossover_point = random.randint(0, parent1.size)
    flat1, flat2 = parent1.flatten(), parent2.flatten()
    child_flat = np.concatenate((flat1[:crossover_point], flat2[crossover_point:]))
    return child_flat.reshape(parent1.shape)

def mutate(individual, mutation_rate):
    num_mutations = int(mutation_rate * individual.size)
    for _ in range(num_mutations):
        x, y, c = random.randint(0, individual.shape[0] - 1), random.randint(0, individual.shape[1] - 1), random.randint(0, 2)
        individual[x, y, c] = random.randint(0, 255)
    return individual

def calculate_mutation_rate(initial_rate, final_rate, current_gen, max_gen=None):
    if max_gen is None:
        decay_rate = (final_rate / initial_rate) ** (1 / current_gen)
    else:
        decay_rate = (final_rate / initial_rate) ** (1 / max_gen)
    return initial_rate * (decay_rate ** current_gen)

def introduce_random_immigrants(population, shape, num_immigrants):
    for _ in range(num_immigrants):
        population[random.randint(0, len(population) - 1)] = create_individual(shape)


def genetic_algorithm(target_image, pop_size=5, initial_mutation_rate=0.0001, final_mutation_rate=0.0001, elitism=3):
    population = create_population(pop_size, target_image.shape)
    best_fitness = float('inf')
    best_individual = None
    generation = 0

    while best_fitness > 5: 
        fitnesses = [calculate_fitness(ind, target_image) for ind in population]

        sorted_population = [ind for _, ind in sorted(zip(fitnesses, population), key=lambda x: x[0])]

        new_population = sorted_population[:elitism]

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2)
            
            mutation_rate = calculate_mutation_rate(initial_mutation_rate, final_mutation_rate, generation, pop_size)
            
            child = mutate(child, mutation_rate)
            new_population.append(child)
        
        if generation % 50 == 0:
            introduce_random_immigrants(new_population, target_image.shape, num_immigrants=pop_size // 10)

        population = new_population

        current_best_fitness = min(fitnesses)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[fitnesses.index(current_best_fitness)]

        if generation % 100 == 0:
            print(f'Generation {generation}, Best Fitness: {best_fitness}')
#           print(f'Generation {generation}, Best Fitness: {best_fitness}')
#           #*****************UNCOMMIT FOR VISUALIZATION IN SEPERATE WINDOW REAL TIME********************
#           #cv2.namedWindow('Reconstructed Image', cv2.WINDOW_NORMAL)
#           #cv2.resizeWindow('Reconstructed Image', 512, 512)
#           #cv2.imshow('Reconstructed Image', best_individual)
#           #if cv2.waitKey(1) & 0xFF == ord('q'):

        if generation % 5000 == 0: #edit to update image every n generations
            yield best_individual

        generation += 1

    return best_individual