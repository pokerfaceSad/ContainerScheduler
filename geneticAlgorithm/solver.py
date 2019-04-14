import numpy as np
import sys
sys.path.append("..")
from environment import *

WITH_GRAPH_OUTPUT = False

'''物品集合'''
global problemSet
global PROBLEM_SIZE
'''箱子大小'''
global BIN_CAPACITY
'''种群规模'''
global POPULATION_SIZE
'''轮盘赌数量'''
global TOURNAMENT_SIZE
'''迭代次数'''
global GENERATIONS
'''适应度函数参数K'''
global K
global SAMPLES
global SAMPLE_RATE
'''突变概率'''
global MUTATION_RATE
'''交叉概率'''
global CROSSOVER_RATE
global organisedChromosome


def solve(serviceChain, serviceLength, binCapacity):
    # 给全局变量赋值
    global problemSet
    global PROBLEM_SIZE
    global BIN_CAPACITY
    global POPULATION_SIZE
    global TOURNAMENT_SIZE
    global GENERATIONS
    global SAMPLE_RATE
    global SAMPLES
    global MUTATION_RATE
    global CROSSOVER_RATE
    global K
    global organisedChromosome

    problemSet = [Environment.service_properties[i]['size'] for i in serviceChain[0:serviceLength]]
    PROBLEM_SIZE = len(problemSet)
    BIN_CAPACITY = binCapacity
    POPULATION_SIZE = 100
    TOURNAMENT_SIZE= 3
    GENERATIONS = 500
    K = 4
    SAMPLES = 1
    SAMPLE_RATE = 50
    MUTATION_RATE = 0.7
    CROSSOVER_RATE = 1
    problemSet = np.array(problemSet)
    # 断言 如果是False会抛出异常
    assert PROBLEM_SIZE == problemSet.size
    organisedChromosome = np.arange(problemSet.size)

    finalFitness = []
    finalBins = []
    bestFitnessOverTime = []
    bestBinsOverTime = []


    for sample in range(SAMPLES):

        # Create new population
        population = []
        position = None
        """
        生成一个随机种群
        """
        chromosome = np.arange(PROBLEM_SIZE)
        for i in range(POPULATION_SIZE):
            np.random.shuffle(chromosome)
            population.append(chromosome.copy())

        bestBinsOverTime.append([])
        bestFitnessOverTime.append([])

        # Mutate and crossover for each generation
        for idx, generation in enumerate(range(GENERATIONS)):

            population = tryMutate(population)
            population = tryCrossover(population)

            """
            每过SAMPLE_RATE代输出一次种群状态
            """
            if idx % SAMPLE_RATE == 0:
                print('GENERATION: ', idx)
                if WITH_GRAPH_OUTPUT:
                    bins = []
                    fitness = []
                    for chromosome in population:
                        binsRequired, chromosomeFitness = getMappedFitness(chromosome)
                        bins.append(binsRequired)
                        fitness.append(chromosomeFitness)

                    position = int(np.argmax(fitness))
                    print('Best in generation: ', bins[position], fitness[position])
                    print()
                    print(problemSet[population[position]])

                    bestFitnessOverTime[sample].append(fitness[position])
                    bestBinsOverTime[sample].append(bins[position])

        """Shit Code"""
        bins = []
        fitness = []
        for chromosome in population:
            binsRequired, chromosomeFitness = getMappedFitness(chromosome)
            bins.append(binsRequired)
            fitness.append(chromosomeFitness)

        position = int(np.argmax(fitness))
        print('Best in generation: ', bins[position], fitness[position])
        print()
        print(problemSet[population[position]])
        bestChromosome = []
        bestChromosome.append(population[position])
        finalFitness.append(fitness[position])
        finalBins.append(bins[position])

    finalFitness = np.array(finalFitness)
    finalBins = np.array(finalBins)

    print('fitness: ')
    print(finalFitness)
    print()
    print('bin num: ')
    print(finalBins)
    if WITH_GRAPH_OUTPUT:

        import matplotlib.pyplot as plt

        iterations = []
        knownBests = []
        for i in range(GENERATIONS):
            if i % SAMPLE_RATE == 0:
                iterations.append(i)
                # knownBests.append(knownBest)


        plt.close()

        fig = plt.figure()
        plt.grid(1)
        plt.xlim([0, GENERATIONS])
        plt.ion()
        plt.xlabel('Generations')
        plt.ylabel('Fitness')

        plots = []
        descriptions = []

        bestFitnessOverTime_mean = np.mean(bestFitnessOverTime, axis=0)
        bestBinsOverTime_mean = np.mean(bestBinsOverTime, axis=0)

        # plots.append(plt.plot(iterations, knownBests, 'ko', alpha=0.2, linewidth=0.5, markersize=3)[0])
        plots.append(plt.plot(iterations, bestFitnessOverTime_mean, 'b--', linewidth=1, markersize=3)[0])
        plots.append(plt.plot(iterations, bestBinsOverTime_mean, 'r-', linewidth=1, markersize=3)[0])
        # descriptions.append("Known best")
        descriptions.append("Best fitness")
        descriptions.append("Bins")

        plt.legend(plots, descriptions)
        fig.savefig('Result.png')
        plt.show()

        plt.close()

        fig, ax = plt.subplots()

    return bestChromosome[0]

"""计算适应度"""
def getMappedFitness(chromosome):
    mappedChromosome = problemSet[chromosome]
    spaces = np.zeros(len(mappedChromosome), dtype=int)
    result = np.cumsum(mappedChromosome) - BIN_CAPACITY
    binsRequired = 0
    totalSpaceLeftOpen = 0
    while True:
        binsRequired += 1
        max_accumulate = np.maximum.accumulate(np.flipud(result <= 0))
        index_of_new_bin = PROBLEM_SIZE - next((idx for idx, val in np.ndenumerate(max_accumulate) if val == True))[0] - 1
        space_left_open = np.abs(result[index_of_new_bin])
        spaces[index_of_new_bin] = space_left_open
        result += space_left_open
        totalSpaceLeftOpen += np.power(((BIN_CAPACITY - space_left_open) / BIN_CAPACITY), K)
        if np.max(result) <= 0:
            break
        result -= BIN_CAPACITY
    return binsRequired, totalSpaceLeftOpen

def toStringMappedFitness(chromosome):
    result = np.cumsum(problemSet[chromosome]) - BIN_CAPACITY
    output = ''
    while True:
        max_accumulate = np.maximum.accumulate(np.flipud(result <= 0))
        index_of_new_bin = PROBLEM_SIZE - next((idx for idx, val in np.ndenumerate(max_accumulate) if val == True))[0] - 1
        space_left_open = np.abs(result[index_of_new_bin])
        result += space_left_open
        output += '|'
        output += (BIN_CAPACITY - space_left_open - 2) * 'X'
        output += '|'
        output += '_' * space_left_open
        output += '\n'
        if np.max(result) <= 0:
            break
        result -= BIN_CAPACITY
    return output

def tournamentSelector(population, reverse=False):
    random_indicies = np.random.randint(POPULATION_SIZE, size=TOURNAMENT_SIZE).tolist()
    tournament = []
    for idx, val in np.ndenumerate(random_indicies):
        tournament.append(population[val])
    results = []
    for val in tournament:
        _, spaces = getMappedFitness(val)
        results.append(spaces)

    results = np.array(results)
    if reverse:
        pos = np.argmin(results)
    else:
        pos = np.argmax(results)
    return population[random_indicies[pos]], random_indicies[pos], results[pos]


def fixChromosome(chrom):
    mask = np.isin(organisedChromosome, chrom, invert=True)
    missing = organisedChromosome[mask].tolist()
    unique, count = np.unique(chrom, return_counts=True)
    duplicated = unique[count > 1].tolist()
    for idx, val in np.ndenumerate(chrom):
        if val in duplicated:
            chrom[idx] = missing.pop()
            duplicated.remove(val)
    return chrom

def multiplePointCrossover(chromosome1, chromosome2, points=4):
    draws = np.random.randint(PROBLEM_SIZE, size=points * 2)
    draws = np.sort(draws)
    c1 = chromosome1.copy()
    c2 = chromosome2.copy()
    current = 0
    for i, val in enumerate(draws):
        c2[current:val], c1[current:val] = chromosome1[current:val], chromosome2[current:val]
        current = val
    c1 = fixChromosome(np.array(c1))
    c2 = fixChromosome(np.array(c2))
    return c1, c2

def multipleSwapMutator(chromosome, swaps=4):
    draws = np.random.randint(PROBLEM_SIZE, size=swaps*2)
    x = draws[:swaps]
    y = draws[swaps:]

    child = chromosome.copy()

    for i, val in enumerate(x):
        tmp = child[val]
        child[val] = child[y[i]]
        child[y[i]] = tmp

    return child

def multipleSwapCrossover(chromosome1, chromosome2, swaps=4):
    draws = np.random.randint(PROBLEM_SIZE, size=swaps)

    c1 = chromosome1.copy()
    c2 = chromosome2.copy()

    for i, val in enumerate(draws):
        c1item = c1[val]
        c2item = c2[val]
        mask1 = np.isin(c2, c1item)
        mask2 = np.isin(c1, c2item)
        c2[mask1] = c2item
        c1[mask2] = c1item
        c1[val] = c2item
        c2[val] = c1item

    return c1, c2


def tryMutate(population):
    draw = np.random.rand()
    if draw < MUTATION_RATE:
        p, pos, fit = tournamentSelector(population)
        _, kpos, _ = tournamentSelector(population, reverse=True)

        c = multipleSwapMutator(p, swaps=np.random.randint(3) + 1)

        population[kpos] = c
    return population


def tryCrossover(population):
    draw = np.random.rand()
    if draw < CROSSOVER_RATE:
        p1, p1pos, p1fit = tournamentSelector(population)
        p2, p2pos, p2fit = tournamentSelector(population)

        if any(p1 != p2):
            _, k1pos, _ = tournamentSelector(population, reverse=True)
            _, k2pos, _ = tournamentSelector(population, reverse=True)

            c1, c2 = multipleSwapCrossover(p1, p2, swaps=np.random.randint(3) + 1)

            population[k1pos] = c1
            population[k2pos] = c2
        else:
            p1 = multipleSwapMutator(p1, swaps=1)

            population[p1pos] = p1

    return population

"""
    将个体转换为position向量
"""
def chrom2position(chromosome):
    serviceLength = len(chromosome)
    position = [0 for _ in range(serviceLength)]
    bin_capacity = BIN_CAPACITY
    bin_no = 0
    for index,service in enumerate(problemSet[chromosome]):
        if service > bin_capacity:
            # 下一个箱子
            bin_no += 1
            bin_capacity = BIN_CAPACITY - service
            position[chromosome[index]] = bin_no
        else:
            bin_capacity -= service
            position[chromosome[index]] = bin_no

    return position

if __name__ == '__main__':
    pass
