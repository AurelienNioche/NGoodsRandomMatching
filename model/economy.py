import numpy as np
from tqdm import tqdm
import itertools as it


class Economy(object):

    def __init__(self, repartition_of_roles, t_max, storing_costs, agent_model,
                 cognitive_parameters=None):

        self.t_max = t_max
        self.cognitive_parameters = cognitive_parameters
        self.storing_costs = storing_costs
        self.agent_model = agent_model

        self.n_goods = len(storing_costs)
        self.roles = self.get_roles(self.n_goods)
        self.repartition_of_roles = np.asarray(repartition_of_roles)
        self.n_agent = sum(self.repartition_of_roles)

        self.agents = None

        # ----- For backup at t ----- #

        self.exchanges = dict()
        for i in it.combinations(range(self.n_goods), r=2):
            self.exchanges[i] = 0
        self.n_exchange = 0
        self.consumption = 0
        self.good_accepted_as_medium = np.zeros(self.n_goods)
        self.proposition_of_medium = np.zeros(self.n_goods)

        # Container for proportions of agents having this or that in hand according to their type
        #  - rows: type of agent
        # - columns: type of good

        self.proportions = np.zeros((self.n_goods, self.n_goods))

        # ---- For final backup ----- #
        self.back_up = {
            "exchanges": [],
            "n_exchanges": [],
            "consumption": [],
            "good_accepted_as_medium": [],
            "proportions": []
        }

    @staticmethod
    def get_roles(n_goods):

        roles = np.zeros((n_goods, 2), dtype=int)

        for i in range(n_goods):
            roles[i] = (i+1) % n_goods, i

        return roles

    def create_agents(self):

        agents = []

        agent_idx = 0

        for agent_type, n in enumerate(self.repartition_of_roles):

            i, j = self.roles[agent_type]

            for ind in range(n):
                a = self.agent_model(
                    prod=i, cons=j,
                    storing_costs=self.storing_costs,
                    cognitive_parameters=self.cognitive_parameters,
                    idx=agent_idx)

                agents.append(a)
                agent_idx += 1

        return agents

    def run(self):

        self.agents = self.create_agents()
        return self.play()

    def play(self):

        for t in tqdm(range(self.t_max)):

            self.time_step()

        return self.back_up

    def time_step(self):

        self.reinitialize_backup_containers()

        self.compute_proportions()

        # ---------- MANAGE EXCHANGES ----- #
        # Take a random order among the indexes of the agents.
        agent_pairs = np.random.choice(self.n_agent, size=(self.n_agent // 2, 2), replace=False)

        for i, j in agent_pairs:
            self.make_encounter(i, j)

        # Each agent consumes at the end of each round and adapt his behavior (or not).
        for agent in self.agents:
            agent.consume()

        self.make_a_backup_for_t()

    def compute_proportions(self):

        # Container for proportions of agents having this or that in hand according to their type
        #  - rows: type of agent
        # - columns: type of good

        for i in self.agents:
            self.proportions[i.C, i.H] += 1  # Type of agent is his consumption good

        for i in range(self.n_goods):
            self.proportions[i] = self.proportions[i] / self.repartition_of_roles[i]

    def make_a_backup_for_t(self):

        # Keep a trace from utilities
        self.consumption = sum([a.consumption for a in self.agents])/self.n_agent

        # ----- FOR FUTURE BACKUP ----- #

        for key in self.exchanges.keys():
            # Avoid division by zero
            if self.n_exchange > 0:
                self.exchanges[key] /= self.n_exchange
            else:
                self.exchanges[key] = 0

        for i in range(self.n_goods):
            # Avoid division by zero
            if self.proposition_of_medium[i] > 0:
                self.good_accepted_as_medium[i] = self.good_accepted_as_medium[i] / self.proposition_of_medium[i]

            else:
                self.good_accepted_as_medium[i] = 0

        assert 0 <= self.good_accepted_as_medium.all() <= 1

        # For back up
        self.back_up["exchanges"].append(self.exchanges.copy())
        self.back_up["consumption"].append(self.consumption)
        self.back_up["n_exchanges"].append(self.n_exchange)
        self.back_up["good_accepted_as_medium"].append(self.good_accepted_as_medium.copy())
        self.back_up["proportions"].append(self.proportions.copy())

    def reinitialize_backup_containers(self):

        # Containers for future backup
        for k in self.exchanges.keys():
            self.exchanges[k] = 0
        self.n_exchange = 0
        self.consumption = 0
        self.good_accepted_as_medium[:] = 0
        self.proposition_of_medium[:] = 0

        self.proportions[:] = 0

    def make_encounter(self, i, j):

        i_H, j_H = self.agents[i].H, self.agents[j].H
        i_P, j_P = self.agents[i].P, self.agents[j].P
        i_C, j_C = self.agents[i].C, self.agents[j].C

        # Each agent is "initiator' of an exchange during one period.
        # Remember that consumption good = type of agent
        i_agreeing = self.agents[i].are_you_satisfied(j_H)
        # is his consumption good
        j_agreeing = self.agents[j].are_you_satisfied(i_H)

        # Consider particular case of offering third object
        i_facing_M = j_H != i_C and i_H == i_P
        j_facing_M = i_H != j_C and j_H == j_P

        # ---- STATS ------ #

        if i_facing_M:
            self.proposition_of_medium[j_H] += 1  # Consider as key the good that is proposed as a medium of ex
            if i_agreeing:
                self.good_accepted_as_medium[j_H] += 1

        if j_facing_M:
            self.proposition_of_medium[i_H] += 1
            if j_agreeing:
                self.good_accepted_as_medium[i_H] += 1

        # ------------ #

        # If both agents agree to exchange...
        if i_agreeing and j_agreeing:

            # ...exchange occurs
            self.agents[i].proceed_to_exchange(j_H)
            self.agents[j].proceed_to_exchange(i_H)

            # ---- STATS ------ #
            exchange_type = tuple(sorted([i_H, j_H]))
            if i_H != j_H:
                self.exchanges[exchange_type] += 1
                self.n_exchange += 1

                # ---------------- #


def launch(**kwargs):
    e = Economy(**kwargs)
    return e.run()
