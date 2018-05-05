import numpy as np
import itertools as it
from model.utils import softmax


class FrequentistAgent(object):

    name = "Frequentist Agent"

    def __init__(self, prod, cons, storing_costs, cognitive_parameters, idx):

        self.P = prod
        self.C = cons
        self.H = self.P

        self.storing_costs = storing_costs

        self.idx = idx

        self.n_goods = len(storing_costs)

        self.consumption = 0

        self.memory_span = cognitive_parameters["memory_span"]
        self.temp = cognitive_parameters["temp"]
        self.u = cognitive_parameters["u"]

        self.encounter = self.get_acceptance_or_encounter_dic(n_goods=self.n_goods)
        self.acceptance = self.get_acceptance_or_encounter_dic(n_goods=self.n_goods)

        self.memory_encounter = self.get_memory_dic(n_goods=self.n_goods)
        self.memory_acceptance = self.get_memory_dic(n_goods=self.n_goods)

        self.in_hand_partner_good_pair = None
        self.accept = None

    @staticmethod
    def get_acceptance_or_encounter_dic(n_goods):

        to_return = dict()

        for i in it.permutations(range(n_goods), r=2):
            to_return[i] = 1.

        return to_return

    @staticmethod
    def get_memory_dic(n_goods):

        memory = dict()
        for i in it.permutations(range(n_goods), r=2):
            memory[i] = []

        return memory

    def are_you_satisfied(self, partner_good):

        self.in_hand_partner_good_pair = self.P, partner_good

        if partner_good == self.C:
            self.accept = 1

        elif partner_good == self.P:
            self.accept = 0

        else:

            self.accept = self.accept_a_medium(partner_good)

        self.learn_from_encounter()

        return self.accept

    def proceed_to_exchange(self, partner_good):

        self.H = partner_good

    def accept_a_medium(self, partner_good):

        v = np.zeros(2)

        # If refuses
        if self.acceptance[(self.P, self.C)] * self.encounter[(self.P, self.C)] > 0:
            v[0] = \
                self.u - self.storing_costs[self.P] / \
                (self.acceptance[(self.P, self.C)] * self.encounter[(self.P, self.C)])
        else:
            v[0] = 0
        # If accepts
        if self.acceptance[(partner_good, self.C)] * self.encounter[(partner_good, self.C)] > 0:
            v[1] = \
                self.u - self.storing_costs[partner_good] / \
                (self.acceptance[(partner_good, self.C)] * self.encounter[(partner_good, self.C)])
        else:
            v[1] = 0

        v = np.tanh(v) * 2 - 1

        p = softmax(v, temp=self.temp)

        return np.random.choice(np.array([0, 1]), p=p)

    def consume(self):

        self.consumption = self.H == self.C

        if self.consumption:
            self.H = self.P

        self.learn_from_result()

    def learn_from_encounter(self):

        for k in self.encounter.keys():
            cond = int(k == self.in_hand_partner_good_pair)
            self.memory_encounter[k].append(cond)
            if len(self.memory_encounter[k]) > self.memory_span:
                self.memory_encounter[k] = self.memory_encounter[k][1:]
            self.encounter[k] = np.mean(self.memory_encounter[k])

    def learn_from_result(self):

        if self.accept:

            successful = int(self.H != self.in_hand_partner_good_pair[0])
            self.memory_acceptance[self.in_hand_partner_good_pair].append(successful)
            if len(self.memory_acceptance[self.in_hand_partner_good_pair]) > self.memory_span:
                self.memory_acceptance[self.in_hand_partner_good_pair] = \
                    self.memory_acceptance[self.in_hand_partner_good_pair][1:]
            self.acceptance[self.in_hand_partner_good_pair] = \
                np.mean(self.memory_acceptance[self.in_hand_partner_good_pair])
