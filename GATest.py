from config import *
from environment import *
from service_batch_generator import *
import sys
sys.path.append(".")
from geneticAlgorithm.solver import *
config, _ = get_config()
config.batch_size = 10
""" Environment """
env = Environment(config.num_bins, config.num_slots, config.num_descriptors)

""" Batch of Services """
services = ServiceBatchGenerator(config.batch_size, config.min_length, config.max_length, config.num_descriptors)

services.getNewState()
servicelengths = services.serviceLength
states = services.state

for index, serviceChain in enumerate(states):
    chromo = solve(serviceChain=serviceChain, serviceLength=servicelengths[index], binCapacity=config.num_slots)
    position = chrom2position(chromo)
    env.step(position, serviceChain, servicelengths[index])
    env.render()
    env.clear()
