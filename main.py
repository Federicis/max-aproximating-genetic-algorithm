import random
from math import ceil, log2
from tabulate import tabulate

# the number of chromosomes
pop_size = None

# function domain
domain = (None, None)

# coefficients of the function(the function is a 2nd degree polynomial
coefficients = (None, None, None)

# domain precision
precision = None

# crossover probability
crossover_probability = None

# mutation probability
mutation_probability = None

# number of iterations
epochs = None

# number of bits in the binary representation of the chromosome
chromosome_length = None

# the max as of now
maxx = None
with open("./data/input.txt", "r") as f:
    pop_size = int(f.readline().strip())
    domain = tuple([float(x.strip()) for x in f.readline().split()])
    coefficients = tuple([float(x.strip()) for x in f.readline().split()])
    precision = int(f.readline().strip())
    crossover_probability = float(f.readline().strip())
    mutation_probability = float(f.readline().strip())
    epochs = int(f.readline().strip())
chromosome_length = ceil(log2((domain[1] - domain[0]) * (10 ** precision)))


def function(x):
    return coefficients[0] * (x ** 2) + coefficients[1] * x + coefficients[2]


# Function to convert Decimal number
# to Binary number
def decimal_to_binary(n):
    binary = []
    val = (n - domain[0]) // ((domain[1] - domain[0]) / (2 ** chromosome_length))
    for i in range(chromosome_length):
        binary.append(str(int(val % 2)))
        val = val // 2
    return ''.join(binary[::-1])


def binary_to_decimal(binary):
    val = 0
    aux = binary[::-1]
    for i in range(chromosome_length):
        if int(aux[i]) == 1:
            val += 2 ** i
    return domain[0] + val * ((domain[1] - domain[0]) / (2 ** chromosome_length))


# binary search
def pick(arr, target):
    low = 0
    high = len(arr) - 1
    result = -1

    while low <= high:
        mid = (low + high) // 2

        if arr[mid] >= target:
            result = mid
            high = mid - 1
        else:
            low = mid + 1

    return result


def print_chromosomes(pop, bin_pop):
    output_table = []
    for i in range(pop_size):
        output_line = ['\t', f'{i + 1}:', f'{bin_pop[i]}', f'x= {pop[i]}', f'f={function(pop[i])}']
        output_table.append(output_line)
    print(tabulate(output_table, tablefmt="plain"), file=output)


current_pop = [random.uniform(round(domain[0], precision), round(domain[1], precision)) for i in range(pop_size)]
current_pop = [round(current_pop[x], precision) for x in range(pop_size)]
binary_pop = list(map(decimal_to_binary, current_pop))

output = open('./data/output.txt', 'w')


def next_epoch(epoch_cnt):
    global maxx
    if epoch_cnt > epochs:
        exit(0)
    if epoch_cnt == 1:
        print('Populatia initiala', file=output)
        print_chromosomes(current_pop, binary_pop)

    suma_totala = sum(list(map(function, current_pop)))
    if epoch_cnt == 1:
        print('\nProbabilitati selectie', file=output)
    output_table = []
    chromosome_probabilities = []
    for i in range(pop_size):
        chromosome_probabilities.append(function(current_pop[i]) / suma_totala)
        if epoch_cnt == 1:
            output_line = ['cromozom', f'{i + 1}:', 'probabilitate', f'{function(current_pop[i]) / suma_totala}']
            output_table.append(output_line)

    if epoch_cnt == 1:
        print(tabulate(output_table, tablefmt="plain"), file=output)
        print('Intervale probabilitati selectie', file=output)

    probability = 0
    cumulative_dist = [0]

    if epoch_cnt == 1:
        print(probability, end=' ', file=output)
    for i in range(len(chromosome_probabilities) - 1):
        probability += float(chromosome_probabilities[i])
        cumulative_dist.append(probability)
        if epoch_cnt == 1:
            print(probability, end=' ', file=output)
    cumulative_dist.append(1)
    if epoch_cnt == 1:
        print(float(1), file=output)

    interm_pop = []
    interm_binary_pop = []
    for i in range(pop_size):
        u = random.random()
        picked_chromosome = pick(cumulative_dist, u)
        if epoch_cnt == 1:
            print(f"u={u}  selectam cromozomul {picked_chromosome}", file=output)
        interm_pop.append(current_pop[picked_chromosome - 1])
        interm_binary_pop.append(binary_pop[picked_chromosome - 1])

    if epoch_cnt == 1:
        print("Dupa selectie:", file=output)
        print_chromosomes(interm_pop, interm_binary_pop)

    crossover_chromosomes = []

    if epoch_cnt == 1:
        print(f"Probabilitatea de incrucisare {crossover_probability}", file=output)
    for i in range(pop_size):
        u = random.random()
        if epoch_cnt == 1:
            print(f"{interm_binary_pop[i]}  u={u}", end="", file=output)
        if u < crossover_probability:
            crossover_chromosomes.append(i + 1)
            if epoch_cnt == 1:
                print(f"<{crossover_probability} participa", file=output)
        else:
            if epoch_cnt == 1:
                print(file=output)

    if len(crossover_chromosomes) % 2 == 1:
        crossover_chromosomes = crossover_chromosomes[:-1]

    for i in range(0, len(crossover_chromosomes), 2):
        if epoch_cnt == 1:
            print(
                f'Recombinare intre cromozomul {crossover_chromosomes[i]} cu cromozomul {crossover_chromosomes[i + 1]}:',
                file=output)
        punct = random.randint(0, 20)
        if epoch_cnt == 1:
            print(
                f'{interm_binary_pop[crossover_chromosomes[i] - 1]} {interm_binary_pop[crossover_chromosomes[i + 1] - 1]} punct {punct}:',
                file=output)
        aux1 = interm_binary_pop[i + 1]
        aux2 = interm_binary_pop[i + 2]
        auxaux = aux2
        aux2 = aux1[:punct] + aux2[punct:]
        aux1 = auxaux[:punct] + aux1[punct:]
        interm_binary_pop[crossover_chromosomes[i] - 1] = aux1
        interm_binary_pop[crossover_chromosomes[i + 1] - 1] = aux2
        interm_pop[crossover_chromosomes[i] - 1] = binary_to_decimal(interm_binary_pop[crossover_chromosomes[i] - 1])
        interm_pop[crossover_chromosomes[i + 1] - 1] = binary_to_decimal(
            interm_binary_pop[crossover_chromosomes[i + 1] - 1])
        if epoch_cnt == 1:
            print(f'Rezultat \t {aux1} {aux2}', file=output)
    if epoch_cnt == 1:
        print(f'Dupa recombinare:', file=output)
        print_chromosomes(interm_pop, interm_binary_pop)
        print(f'Probabilitati de mutatie pentru fiecare gena {mutation_probability}', file=output)
    mutation_chromosomes = []
    for i in range(0, pop_size):
        u = random.random()
        if u < mutation_probability:
            mutation_chromosomes.append(i + 1)

    for i in range(0, len(mutation_chromosomes)):
        punct = random.randint(0, 20)
        bit = 1 - int(interm_binary_pop[mutation_chromosomes[i] - 1][punct])
        interm_binary_pop[mutation_chromosomes[i] - 1] = interm_binary_pop[mutation_chromosomes[i] - 1][:punct] + str(
            bit) + interm_binary_pop[mutation_chromosomes[i] - 1][punct + 1:]
        interm_pop[mutation_chromosomes[i] - 1] = binary_to_decimal(interm_binary_pop[mutation_chromosomes[i] - 1])
    if len(mutation_chromosomes) > 0 and epoch_cnt == 1:
        print("Au fost modificati cromozomii:", file=output)
        for i in range(0, len(mutation_chromosomes)):
            print(mutation_chromosomes[i], file=output)

    if epoch_cnt == 1:
        print("Dupa mutatie:", file=output)
        print_chromosomes(interm_pop, interm_binary_pop)

    for i in range(pop_size):
        if maxx is None or function(interm_pop[i]) > maxx:
            maxx = function(interm_pop[i])
    if epoch_cnt == 1:
        print(f"Evolutia maximului:", file=output)
    print(maxx, file=output)
    next_epoch(epoch_cnt + 1)

next_epoch(1)