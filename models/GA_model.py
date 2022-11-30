import copy
import numpy as np
import sys
sys.path.append('/root/neuro_evolution')#我表示我不理解为什么直接调不了可能是我服务器的问题，先这么放着吧
from utils.tools import model_weights_as_vector
class TorchGA:
    def __init__(self, model, num_solutions):

        """
        Creates an instance of the TorchGA class to build a population of model parameters.
        model: A PyTorch model class.
        num_solutions: Number of solutions in the population. Each solution has different model parameters.
        """

        self.model = model
        self.num_solutions = num_solutions
        # A list holding references to all the solutions (i.e. networks) used in the population.
        self.population_weights = self.create_population()

    def create_population(self):

        """
        Creates the initial population of the genetic algorithm as a list of networks' weights (i.e. solutions). Each element in the list holds a different weights of the PyTorch model.
        The method returns a list holding the weights of all solutions.
        """

        model_weights_vector = model_weights_as_vector(model=self.model)

        net_population_weights = []
        net_population_weights.append(model_weights_vector)

        for idx in range(self.num_solutions - 1):
            net_weights = copy.deepcopy(model_weights_vector)
            net_weights = np.array(net_weights) + np.random.uniform(low=-1.0, high=1.0,
                                                                          size=model_weights_vector.size)

            # Appending the weights to the population.
            net_population_weights.append(net_weights)

        return net_population_weights