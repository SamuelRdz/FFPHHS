import random
import math
import re
import pandas as pd
from os import listdir
from os.path import isfile, join
# Provides the methods to create and solve the firefighter problem
from PBC4cip import *
from PBC4cip.core.Helpers import get_col_dist, get_idx_val


class FFP:

    # Constructor
    #   fileName = The name of the file that contains the FFP instance
    def __init__(self, fileName):
        file = open(fileName, "r")
        text = file.read()
        tokens = text.split()
        seed = int(tokens.pop(0))
        self.n = int(tokens.pop(0))
        model = int(tokens.pop(0))
        int(tokens.pop(0))  # Ignored
        # self.state contains the state of each node
        #    -1 On fire
        #     0 Available for analysis
        #     1 Protected
        self.state = [0] * self.n
        nbBurning = int(tokens.pop(0))
        for i in range(nbBurning):
            b = int(tokens.pop(0))
            self.state[b] = -1
        self.graph = []
        for i in range(self.n):
            self.graph.append([0] * self.n)
        while tokens:
            x = int(tokens.pop(0))
            y = int(tokens.pop(0))
            self.graph[x][y] = 1
            self.graph[y][x] = 1

    # Solves the FFP by using a given method and a number of firefighters
    #   method = Either a string with the name of one available heuristic or an object of class HyperHeuristic
    #   nbFighters = The number of available firefighters per turn
    #   debug = A flag to indicate if debugging messages are shown or not
    def solve(self, method, nbFighters, debug=False):
        spreading = True
        if (debug):
            print("Initial state:" + str(self.state))
        t = 0
        while (spreading):
            if (debug):
                print("Features")
                print("")
                print("Graph density: %1.4f" %
                      (self.getFeature("EDGE_DENSITY")))
                print("Average degree: %1.4f" %
                      (self.getFeature("AVG_DEGREE")))
                print("Burning nodes: %1.4f" %
                      self.getFeature("BURNING_NODES"))
                print("Burning edges: %1.4f" %
                      self.getFeature("BURNING_EDGES"))
                print("Nodes in danger: %1.4f" %
                      self.getFeature("NODES_IN_DANGER"))
            # It protects the nodes (based on the number of available firefighters)
            for i in range(nbFighters):
                heuristic = method
                if (isinstance(method, HyperHeuristic)):
                    heuristic = method.nextHeuristic(self)
                node = self.__nextNode(heuristic)
                if (node >= 0):
                    # The node is protected
                    self.state[node] = 1
                    # The node is disconnected from the rest of the graph
                    for j in range(len(self.graph[node])):
                        self.graph[node][j] = 0
                        self.graph[j][node] = 0
                    if (debug):
                        print("\tt" + str(t) +
                              ": A firefighter protects node " + str(node))
            # It spreads the fire among the unprotected nodes
            spreading = False
            state = self.state.copy()
            for i in range(len(state)):
                # If the node is on fire, the fire propagates among its neighbors
                if (state[i] == -1):
                    for j in range(len(self.graph[i])):
                        if (self.graph[i][j] == 1 and state[j] == 0):
                            spreading = True
                            # The neighbor is also on fire
                            self.state[j] = -1
                            # The edge between the nodes is removed (it will no longer be used)
                            self.graph[i][j] = 0
                            self.graph[j][i] = 0
                            if (debug):
                                print("\tt" + str(t) +
                                      ": Fire spreads to node " + str(j))
            t = t + 1
            if (debug):
                print("---------------")
        if (debug):
            print("Final state: " + str(self.state))
            print("Solution evaluation: " +
                  str(self.getFeature("BURNING_NODES")))
        return self.getFeature("BURNING_NODES")

    # Selects the next node to protect by a firefighter
    #   heuristic = A string with the name of one available heuristic
    def __nextNode(self, heuristic):
        index = -1
        best = -1
        for i in range(len(self.state)):
            if (self.state[i] == 0):
                index = i
                break
        value = -1
        for i in range(len(self.state)):
            if (self.state[i] == 0):
                if (heuristic == "LDEG"):
                    # It prefers the node with the largest degree, but it only considers
                    # the nodes directly connected to a node on fire
                    for j in range(len(self.graph[i])):
                        if (self.graph[i][j] == 1 and self.state[j] == -1):
                            value = sum(self.graph[i])
                            break
                elif (heuristic == "GDEG"):
                    value = sum(self.graph[i])
                else:
                    print("=====================")
                    print("Critical error at FFP.__nextNode.")
                    print("Heuristic " + heuristic +
                          " is not recognized by the system.")
                    print("The system will halt.")
                    print("=====================")
                    exit(0)
            if (value > best):
                best = value
                index = i
        return index

    # Returns the value of the feature provided as argument
    #   feature = A string with the name of one available feature
    def getFeature(self, feature):
        f = 0
        if (feature == "EDGE_DENSITY"):
            n = len(self.graph)
            for i in range(len(self.graph)):
                f = f + sum(self.graph[i])
            f = f / (n * (n - 1))
        elif (feature == "AVG_DEGREE"):
            n = len(self.graph)
            count = 0
            for i in range(len(self.state)):
                if (self.state[i] == 0):
                    f += sum(self.graph[i])
                    count += 1
            if (count > 0):
                f /= count
                f /= (n - 1)
            else:
                f = 0
        elif (feature == "BURNING_NODES"):
            for i in range(len(self.state)):
                if (self.state[i] == -1):
                    f += 1
            f = f / len(self.state)
        elif (feature == "BURNING_EDGES"):
            n = len(self.graph)
            for i in range(len(self.graph)):
                for j in range(len(self.graph[i])):
                    if (self.state[i] == -1 and self.graph[i][j] == 1):
                        f += 1
            f = f / (n * (n - 1))
        elif (feature == "NODES_IN_DANGER"):
            for j in range(len(self.state)):
                for i in range(len(self.state)):
                    if (self.state[i] == -1 and self.graph[i][j] == 1):
                        f += 1
                        break
            f /= len(self.state)
        else:
            print("=====================")
            print("Critical error at FFP._getFeature.")
            print("Feature " + feature + " is not recognized by the system.")
            print("The system will halt.")
            print("=====================")
            exit(0)
        return f

    # Returns the string representation of this problem
    def __str__(self):
        text = "n = " + str(self.n) + "\n"
        text += "state = " + str(self.state) + "\n"
        for i in range(self.n):
            for j in range(self.n):
                if (self.graph[i][j] == 1 and i < j):
                    text += "\t" + str(i) + " - " + str(j) + "\n"
        return text

# Provides the methods to create and use hyper-heuristics for the FFP
# This is a class you must extend it to provide the actual implementation


class HyperHeuristic:

    # Constructor
    #   features = A list with the names of the features to be used by this hyper-heuristic
    #   heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
    def __init__(self, features, heuristics):
        if (features):
            self.features = features.copy()
        else:
            print("=====================")
            print("Critical error at HyperHeuristic.__init__.")
            print("The list of features cannot be empty.")
            print("The system will halt.")
            print("=====================")
            exit(0)
        if (heuristics):
            self.heuristics = heuristics.copy()
        else:
            print("=====================")
            print("Critical error at HyperHeuristic.__init__.")
            print("The list of heuristics cannot be empty.")
            print("The system will halt.")
            print("=====================")
            exit(0)

    # Returns the next heuristic to use
    #   problem = The FFP instance being solved
    def nextHeuristic(self, problem):
        print("=====================")
        print("Critical error at HyperHeuristic.nextHeuristic.")
        print("The method has not been overriden by a valid subclass.")
        print("The system will halt.")
        print("=====================")
        exit(0)

    # Returns the string representation of this hyper-heuristic
    def __str__(self):
        print("=====================")
        print("Critical error at HyperHeuristic.__str__.")
        print("The method has not been overriden by a valid subclass.")
        print("The system will halt.")
        print("=====================")
        exit(0)

# A dummy hyper-heuristic for testing purposes.
# The hyper-heuristic creates a set of randomly initialized rules.
# Then, when called, it measures the distance between the current state and the
# conditions in the rules
# The rule with the condition closest to the problem state is the one that fires


class DummyHyperHeuristic(HyperHeuristic):

    # Constructor
    #   features = A list with the names of the features to be used by this hyper-heuristic
    #   heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
    #   nbRules = The number of rules to be contained in this hyper-heuristic
    def __init__(self, features, heuristics, nbRules, seed):
        super().__init__(features, heuristics)
        random.seed(seed)
        self.conditions = []
        self.actions = []
        for i in range(nbRules):
            self.conditions.append([0] * len(features))
            for j in range(len(features)):
                self.conditions[i][j] = random.random()
            self.actions.append(
                heuristics[random.randint(0, len(heuristics) - 1)])

    # Returns the next heuristic to use
    #   problem = The FFP instance being solved
    def nextHeuristic(self, problem):
        minDistance = float("inf")
        index = -1
        state = []
        for i in range(len(self.features)):
            state.append(problem.getFeature(self.features[i]))
        print("\t" + str(state))
        for i in range(len(self.conditions)):
            distance = self.__distance(self.conditions[i], state)
            if (distance < minDistance):
                minDistance = distance
                index = i
        heuristic = self.actions[index]
        print("\t\t=> " + str(heuristic) + " (R" + str(index) + ")")
        return heuristic

    # Returns the string representation of this dummy hyper-heuristic
    def __str__(self):
        text = "Features:\n\t" + \
            str(self.features) + "\nHeuristics:\n\t" + \
            str(self.heuristics) + "\nRules:\n"
        for i in range(len(self.conditions)):
            text += "\t" + \
                str(self.conditions[i]) + " => " + self.actions[i] + "\n"
        return text

    # Returns the Euclidian distance between two vectors
    def __distance(self, vectorA, vectorB):
        distance = 0
        for i in range(len(vectorA)):
            distance += (vectorA[i] - vectorB[i]) ** 2
        distance = math.sqrt(distance)
        return distance


class PBC4cipHyperHeuristic(HyperHeuristic):

    # Constructor
    #   features = A list with the names of the features to be used by this hyper-heuristic
    #   heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
    #   nbRules = The number of rules to be contained in this hyper-heuristic
    def __init__(self, features, heuristics, train_file, seed):
        super().__init__(features, heuristics)
        random.seed(seed)
        self.train_file = train_file
        self.train_PBC4cip()

    def train_PBC4cip(self):
        df = pd.read_csv(self.train_file)
        # df = df.sample(500)
        # print(df)
        self.X_train = df[self.features]
        self.y_train = df[["HEURISTIC"]]
        self.clf = PBC4cip(filtering=False)
        patterns = self.clf.fit(self.X_train, self.y_train)
        print(f"\nPatterns Found:")
        f = open("patterns.txt", "w")
        for pattern in patterns:
            print(f"{pattern}", file=f)
        self.actions = get_col_dist(
            self.y_train[f'{self.y_train.columns[0]}'])

    # Returns the next heuristic to use
    #   problem = The FFP instance being solved
    def nextHeuristic(self, problem):
        state = []
        for i in range(len(self.features)):
            state.append(problem.getFeature(self.features[i]))
        test = pd.DataFrame(columns=self.features, data=[state])
        y_pred = self.clf.predict(test)  # predict PBC4cip
        heuristic = self.actions[y_pred[0]]
        print("\t" + str(state) + "\t=> " + str(heuristic))
        return heuristic

    # Returns the string representation of this dummy hyper-heuristic
    def __str__(self):
        text = "Features:\n\t" + \
            str(self.features) + "\nHeuristics:\n\t" + \
            str(self.heuristics)
        return text


class RandomHyperHeuristic(HyperHeuristic):

    # Constructor
    #   features = A list with the names of the features to be used by this hyper-heuristic
    #   heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
    #   nbRules = The number of rules to be contained in this hyper-heuristic
    def __init__(self, features, heuristics, seed):
        super().__init__(features, heuristics)
        random.seed(seed)
        self.conditions = []
        self.actions = heuristics.copy()
        self.solution = []

    # Returns the next heuristic to use
    #   problem = The FFP instance being solved
    def nextHeuristic(self, problem):
        minDistance = float("inf")
        index = -1
        state = []
        for i in range(len(self.features)):
            state.append(problem.getFeature(self.features[i]))
        # print("\t" + str(state))
        heuristic = self.actions[random.randint(0, len(self.actions) - 1)]
        # print("\t\t=> " + str(heuristic))
        state.append(heuristic)
        self.solution.append(state)
        return heuristic

    # Returns the string representation of this dummy hyper-heuristic
    def __str__(self):
        text = "Features:\n\t" + \
            str(self.features) + "\nHeuristics:\n\t" + \
            str(self.heuristics) + "\nRules:\n"
        for i in range(len(self.conditions)):
            text += "\t" + \
                str(self.conditions[i]) + " => " + self.actions[i] + "\n"
        return text

    # Returns the Euclidian distance between two vectors
    def __distance(self, vectorA, vectorB):
        distance = 0
        for i in range(len(vectorA)):
            distance += (vectorA[i] - vectorB[i]) ** 2
        distance = math.sqrt(distance)
        return distance


# fileName = "instances/BBGRL/0prueba.in"
# fileName = "instances/BBGRL/50_ep0.2_0_gilbert_1.in"
# Solves the problem using heuristic LDEG and one firefighter
# problem = FFP(fileName)
# print("LDEG = " + str(problem.solve("LDEG", 1, False)))

# Solves the problem using heuristic GDEG and one firefighter
# problem = FFP(fileName)
# print("GDEG = " + str(problem.solve("GDEG", 1, False)))

# Solves the problem using a randomly generated dummy hyper-heuristic
# problem = FFP(fileName)
# seed = random.randint(0, 1000)
# print(seed)
# hh = DummyHyperHeuristic(["EDGE_DENSITY", "BURNING_NODES", "NODES_IN_DANGER"], [
#                          "LDEG", "GDEG"], 2, seed)
# print(hh)
# print("Dummy HH = " + str(problem.solve(hh, 1, False)))
# features = ["EDGE_DENSITY", "AVG_DEGREE",
#             "BURNING_NODES", "BURNING_EDGES", "NODES_IN_DANGER"]
# pbc_hh = PBC4cipHyperHeuristic(
#     features, ["LDEG", "GDEG"], 'train_set.csv', seed)
# problem = FFP(fileName)
# result = problem.solve(pbc_hh, 1, False)
# print("PBC4cip HH = " + str(result))


# rhh = RandomHyperHeuristic(features, ["LDEG", "GDEG"], seed)
# result = problem.solve(rhh, 1, False)
# # print("Random HH = " + str(result))
# # print('Solution \n', rhh.solution)
# df = pd.DataFrame(columns=features + ["HEURISTIC"], data=rhh.solution)
# df.insert(0, 'INSTANCE', fileName)
# df['RESULT'] = result
# print(df)

def dataFilenames():

    bbgrlPath = 'instances/BBGRL'
    ggrlPath = 'instances/GBRL'

    bbgrlInstances = [bbgrlPath+'/' +
                      f for f in listdir(bbgrlPath) if isfile(join(bbgrlPath, f))]
    gbrlInstances = [ggrlPath+'/' +
                     f for f in listdir(ggrlPath) if isfile(join(ggrlPath, f))]

    random.seed(42)
    random.shuffle(bbgrlInstances)
    random.shuffle(gbrlInstances)

    training_bbgrl = bbgrlInstances[:(len(bbgrlInstances) // 10) * 8]
    testing_bbgrl = bbgrlInstances[(len(bbgrlInstances) // 10) * 8:]

    training_ggrl = gbrlInstances[:(len(gbrlInstances) // 10) * 8]
    testing_ggrl = gbrlInstances[(len(gbrlInstances) // 10) * 8:]

    training = training_bbgrl + training_ggrl
    testing = testing_bbgrl + testing_ggrl

    return(training, testing)


def create_train_data(file_names=[]):
    features = ["EDGE_DENSITY", "AVG_DEGREE",
                "BURNING_NODES", "BURNING_EDGES", "NODES_IN_DANGER"]
    final_train_set = pd.DataFrame()
    j = 0
    for file_name in file_names:
        j += 1
        print(j, file_name)
        problem = FFP(file_name)
        n = problem.n
        best_result = 1
        best_solution = pd.DataFrame()
        for i in range(10):
            problem = FFP(file_name)
            seed = random.randint(0, 1000)
            rhh = RandomHyperHeuristic(features, ["LDEG", "GDEG"], seed)
            result = problem.solve(rhh, 1, False)
            df = pd.DataFrame(columns=features +
                              ["HEURISTIC"], data=rhh.solution)
            df.insert(0, 'INSTANCE', file_name)
            df['RESULT'] = result
            # print(df)
            if result < best_result:
                best_result = result
                best_solution = df
        if final_train_set.empty:
            final_train_set = best_solution
        else:
            final_train_set = pd.concat([final_train_set, best_solution])
            # final_train_set.append(best_solution)
    print('FINAL TRAIN SET \n', final_train_set)
    final_train_set.to_csv('train_set.csv')


# file_names = [fileName]
# file_names = ['instances/BBGRL/1000_ep0.0125_0_gilbert_4.in']

# file_names, test_set = dataFilenames()
# create_train_data(file_names)

# Tests
# =====================
def test():
    file_names, test_set = dataFilenames()
    seed = random.randint(0, 1000)
    features = ["EDGE_DENSITY", "AVG_DEGREE",
                "BURNING_NODES", "BURNING_EDGES", "NODES_IN_DANGER"]
    pbc_hh = PBC4cipHyperHeuristic(
        features, ["LDEG", "GDEG"], 'train_set.csv', seed)
    instances = []
    ldeg_results = []
    gdeg_results = []
    oracle_results = []
    pbc_results = []
    for file_name in test_set:
        instances.append(file_name)
        print('Instance:', file_name)
        problem = FFP(file_name)
        LDEG = problem.solve("LDEG", 1, False)
        ldeg_results.append(LDEG)
        print("LDEG = " + str(LDEG))
        problem = FFP(file_name)
        GDEG = problem.solve("GDEG", 1, False)
        gdeg_results.append(GDEG)
        print("GDEG = " + str(GDEG))
        oracle_results.append(min(LDEG, GDEG))
        problem = FFP(file_name)
        PBC = problem.solve(pbc_hh, 1, False)
        pbc_results.append(PBC)
        print("PBC4cip HH = " + str(PBC))
    final_df = pd.DataFrame(data={
        'INSTANCE': instances, 'LDEG': ldeg_results, 'GDEG': gdeg_results, 'Oracle': oracle_results, 'PBC4cip': pbc_results
    })
    final_df.to_csv('final_df.csv')


test()
