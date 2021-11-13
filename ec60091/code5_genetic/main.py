import random
import matplotlib.pyplot as plt

def cal_fitness_score(chromo):
    """
    Function to calculate the fitness score of a chromosome
    """
    ans = 0.0
    for chr in chromo:
        ans += (chr == '1')
    return ans

def generate_chromosome():
    """
    Function to generate chromosome, with 1 and 0 equiprobable
    """
    s = ""
    for i in range(20):
        if random.randint(0,1):
            s += '1'
        else:
            s += '0'
    return s

def crossover(parent1, parent2):
    """
    Function for mating and generating
    """
    crossover_pt = random.randint(0,20)
    child_1 = parent1[:crossover_pt] + parent2[crossover_pt:]
    child_2 = parent2[:crossover_pt] + parent1[crossover_pt:]
    child_3 = ""
    child_4 = ""
    for i in range(20):
        x = random.randint(0,1)
        if x:
            child_3 += parent1[i]
            child_4 += parent2[i]
        else:
            child_3 += parent2[i]
            child_4 += parent1[i]
    return child_1, child_2, child_3, child_4

def mutate_chromosome(chromo):
    """
    Function for mutation for a 1% mutation rate
    """
    ans = ""
    for chr in chromo:
        x = random.randint(0,99)
        if not x:
            if chr == '0':
                ans += '1'
            else:
                ans += '0'
        else:
            ans += chr
    
    return ans

def best(pop):
    """
    Select the top 100 fitness scores
    """
    scores = [(cal_fitness_score(chromo), idx) \
                for idx, chromo in enumerate(pop)]
    scores.sort(reverse=True)
    best_pop = [pop[scores[i][1]] for i in range(100)]
    
    return best_pop

def new_population(old_population):
    old_population = best(old_population)
    idxs = list(range(100))
    new_population = []
    random.shuffle(idxs)
    for i in range(50):
        idx1 = 2*i
        idx2 = 2*i + 1
        parent1, parent2 = old_population[idx1], old_population[idx2]
        children = crossover(parent1, parent2)
        for i in range(4): new_population.append(children[i])
    
    return new_population

if __name__ == '__main__':
    # Generate a starting population
    population = [generate_chromosome() for i in range(200)]

    maxs = []   # Lists to maintain metrics
    avgs = []

    for itr in range(300):
        scores = [cal_fitness_score(chromo) for chromo in population]
        best_score = max(scores)
        best_idx = scores.index(best_score)
        mean_score = (1.0*sum(scores))/len(scores)
        avgs.append(mean_score)
        maxs.append(best_score)

        print("Best chromosome for generation %d is %s, with fitness score %.3f" % (itr+1, population[best_idx], best_score))

        population = new_population(population)
        population = [mutate_chromosome(chromo) for chromo in population]

    scores = [cal_fitness_score(chromo) for chromo in population]
    best_score = max(scores)
    best_idx = scores.index(best_score)
    mean_score = (1.0*sum(scores))/len(scores)

    print("Best chromosome after last generation is %s, with fitness score %.3f" % (population[best_idx], best_score))

    #Plot average and maximum fitness for each generation
    plt.figure()
    plt.plot(range(len(avgs)), avgs)
    plt.ylabel('Average Fitness')
    plt.xlabel('Generation')
    plt.title('Average Fitness for each generation')
    plt.savefig('avg.png')

    plt.figure()
    plt.plot(range(len(maxs)), maxs)
    plt.ylabel('Maximum Fitness')
    plt.xlabel('Generation')
    plt.title('Maximum Fitness for each generation')
    plt.savefig('max.png')