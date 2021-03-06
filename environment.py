#-*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Environment(object):
    """
        Implementation of the black-boxed environment

        Attributes common to the environment:
            numBins(int) -- Number of bins in the environment
            numSlots(int) -- Number of available slots per bin in the environment
            cells[numBins, numSlots] -- Stores environment occupancy
            packet_properties{struct} -- Describes packet properties inside the environment

        Attributes common to the service
            serviceLength(int) -- Length of the service
            service[serviceLength] -- Collects the service chain
            placement[serviceLength] -- Collects the packet allocation for the service chain
            first_slots[serviceLength] -- Stores the first slot occupied in the correspondent bin for each packet
            reward(float) -- Stores the reward obtained placing the service on the environment
            invalidPlacement(Bool) -- Invalid placement indicates that there is a resource overflow
    """
    # 为方便GA调用将service_properties声明为类变量
    service_properties = [{"size": 1} for _ in range(20)]
    service_properties[0]["size"] = 1
    service_properties[1]["size"] = 2
    service_properties[2]["size"] = 4
    service_properties[3]["size"] = 8
    service_properties[4]["size"] = 16
    service_properties[5]["size"] = 32
    service_properties[6]["size"] = 64

    def __init__(self, numBins, numSlots, numDescriptors):

        # Environment properties
        self.numBins = numBins
        self.numSlots = numSlots
        self.numDescriptors = numDescriptors
        self.cells = np.empty((numBins, numSlots))
        self.cells[:] = np.nan
        self.service_properties = [{"size": 1} for _ in range(numDescriptors)]

        # Placement properties
        self.serviceLength = 0
        self.service = None
        self.placement = None
        self.first_slots = None
        self.reward = 1
        self.invalidPlacement = False

        # Assign ns properties within the environment
        self._get_service_propertieses()

    def _get_service_propertieses(self):
        """ Packet properties """
        # By default the size of each package in that environment is 1, should be modified here.
        self.service_properties[0]["size"] = 1
        self.service_properties[1]["size"] = 2
        self.service_properties[2]["size"] = 4
        self.service_properties[3]["size"] = 8
        self.service_properties[4]["size"] = 16
        self.service_properties[5]["size"] = 32
        self.service_properties[6]["size"] = 64

    def _placeSubPakcet(self, bin, pkt):
        """ Place subPacket """

        occupied_slot = None
        for slot in range(len(self.cells[bin])):
            if np.isnan(self.cells[bin][slot]):
                self.cells[bin][slot] = pkt
                occupied_slot = slot
                break
            elif slot == len(self.cells[bin])-1:
                self.invalidPlacement = True
                occupied_slot = -1      # No space available
                break
            else:
                pass                    # Look for next slot

        return occupied_slot

    def _placePacket(self, i, bin, pkt):
        """ Place Packet """

        for slot in range(self.service_properties[pkt]["size"]):
            occupied_slot = self._placeSubPakcet(bin, pkt)

            # Anotate first slot used by the Packet
            if slot == 0:
                self.first_slots[i] = occupied_slot

    def _computeReward(self):
        """ Compute reward """

        occupancy = np.empty(self.numBins)
        for bin in range(self.numBins):
            occupied = 0
            for slot in range(len(self.cells[bin])):
                if not math.isnan(self.cells[bin][slot]):
                    occupied += 1

            occupancy[bin] = occupied / len(self.cells[bin])

        # reward = np.sum(np.power(100, occupancy))
        size = np.flatnonzero(occupancy).size         
        reward = np.sum(occupancy)/size
        return reward

    def step(self, placement, service, length):
        """ Place service """

        self.placement = placement
        self.service = service
        self.serviceLength = length
        self.first_slots = np.zeros(length, dtype='int32')

        for i in range(length):
            self._placePacket(i, placement[i], service[i])

        """ Compute reward """
        if self.invalidPlacement == True:
            self.reward = -1
        else:
            self.reward = self._computeReward()

    def clear(self):
        """ Clean environment """

        self.cells = np.empty((self.numBins, self.numSlots))
        self.cells[:] = np.nan
        self.serviceLength = 0
        self.service = None
        self.placement = None
        self.first_slots = None
        self.reward = -1
        self.invalidPlacement = False

    def render(self, epoch=0):
        """ Render environment using Matplotlib """

        # Creates just a figure and only one subplot
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.set_title(f'Environment {epoch}\nreward: {self.reward}')

        margin = 1
        margin_ext = 2
        xlim = 400
        ylim = 80

        # Set drawing limits
        plt.xlim(0, xlim)
        plt.ylim(-ylim, 0)

        # Set hight and width for the box
        high = np.floor((ylim - 2 * margin_ext - margin * (self.numBins - 1)) / self.numBins)
        wide = np.floor((xlim - 2 * margin_ext - margin * (self.numSlots - 1)) / self.numSlots)

        # Plot slot labels
        # for slot in range(self.numSlots):
        #     x = wide * slot + slot * margin + margin_ext
        #     plt.text(x + 0.5 * wide, -3, "slot{}".format(slot), ha="center", family='sans-serif', size=8)

        # Plot bin labels & place empty boxes
        for bin in range(self.numBins):
            y = -high * (bin + 1) - (bin) * margin - margin_ext
            plt.text(0, y + 0.5 * high, "bin{}".format(bin), ha="center", family='sans-serif', size=8)

            for slot in range(self.numSlots):
                x = wide * slot + slot * margin + margin_ext
                rectangle = mpatches.Rectangle((x, y), wide, high, linewidth=1, edgecolor='black', facecolor='none')
                ax.add_patch(rectangle)

        # Select serviceLength colors from a colormap
        cmap = plt.cm.get_cmap('hot')
        colormap = [cmap(np.float32(i+1)/(self.serviceLength+1)) for i in range(self.serviceLength)]

        # Plot service boxes
        for idx in range(self.serviceLength):
            pkt = self.service[idx]
            bin = self.placement[idx]
            first_slot = self.first_slots[idx]

            for k in range(self.service_properties[pkt]["size"]):
                slot = first_slot + k
                x = wide * slot + slot * margin + margin_ext
                y = -high * (bin + 1) - bin * margin - margin_ext
                rectangle = mpatches.Rectangle((x, y), wide, high, linewidth=0, facecolor=colormap[idx], alpha=.9)
                ax.add_patch(rectangle)
                if k == 1:
                    plt.text(x + 0.5 * wide, y + 0.5 * high, "pkt{}".format(pkt), ha="center", family='sans-serif', size=8)



        plt.axis('off')
        plt.show()


if __name__ == "__main__":

    # Define environment
    numBins = 5
    numSlots = 128
    numDescriptors = 7
    env = Environment(numBins, numSlots, numDescriptors)

    # Allocate service in the environment
    servicelength = 5
    ns = [0, 6, 6, 5, 0]
    import sys
    sys.path.append(".")
    from geneticAlgorithm.solver import *
    chrom = solve(ns, servicelength, numSlots)
    position = chrom2position(chrom)
    # placement = [0, 1, 1, 0, 0]
    env.step(position, ns, servicelength)
    env.render()
    env.clear()
