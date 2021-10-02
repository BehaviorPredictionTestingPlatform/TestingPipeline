"""Simulated annealing samplers"""

from verifai.samplers.domain_sampler import BoxSampler, SamplingError
import math
import numpy as np

def proposal_func(sample, iteration, decay_rate, num_variables):

    decayed_width = decay_rate**iteration*1
    lower_bound = [(sample[i] - decayed_width/2) for i in range(num_variables)]
    upper_bound = [(sample[i] + decayed_width/2) for i in range(num_variables)]

    # check whether updated range for each parameter is within original ranges
    for index in range(num_variables):
        if lower_bound[index] < 0:
            lower_bound[index] = 0
        if upper_bound[index] > 1:
            upper_bound[index] = 1

    return np.random.uniform(lower_bound, upper_bound)


class SimulatedAnnealingSampler(BoxSampler):
    def __init__(self, domain, sa_params):
        super().__init__(domain)

        ## Initialize three functions :
        if sa_params.temp_f is None :
            self.temp_f = lambda t: 0.8*t
        else :
            self.temp_f = sa_params.temp_f

        if sa_params.iter_f is None :
            self.iter_f = lambda length: int(math.ceil(1.1*length))
        else :
            self.iter_f = sa_params.iter_f

        if sa_params.proposal_f is None :
            self.proposal_f = proposal_func
        else :
            self.proposal_f = sa_params.proposal_f


        ## Initialize Parameters :
        if sa_params.reset_temp is None :
            self.reset_temp = 0.01
        else :
            self.reset_temp = sa_params.reset_temp

        
        self.T = self.init_T = sa_params.T
        self.decay_rate = sa_params.decay_rate
        self.iterations = sa_params.iterations
        self.num_epoch = sa_params.num_epoch
        self.num_iter_in_epoch = 0
        self.old_sample = self.best_sample = None
        self.old_loss = None

    def nextVector(self, feedback=None):
        # num_epoch := total # of rise of temp
        if self.num_epoch <= 0:
            raise SamplingError("Total Number of Epochs Executed")

        if feedback is None:    # First sample
            assert self.old_sample is None
            sample = np.random.uniform(0,1, self.dimension)
            self.old_sample = sample
            self.old_loss = None

            ### Update Parameters before Returning ###
            # update temperature
            self.T = self.temp_f(self.T)

            # update iteration count within an epoch and total iteration count
            self.num_iter_in_epoch += 1

            return sample

        if self.old_loss is None:
            self.old_loss = feedback
        else:
            ## Compute the loss of the sample
            new_loss = feedback

            # Compute the probability of whether this sample should be accepted
            alpha = min(1, np.exp((self.old_loss - new_loss)/self.T))

            ## Note that here objective is to MINIMIZE the loss
            if ((new_loss < self.old_loss) or (np.random.uniform() < alpha)):
                # Accept proposed solution
                self.old_loss = new_loss
                self.old_sample = self.last_sample

            ### Update Parameters before Returning ###
            # update temperature
            self.T = self.temp_f(self.T)

            # update iteration count within an epoch and total iteration count
            self.num_iter_in_epoch += 1

            ### Check for Temperature cooling below a threshold or
            ### Executing given # of iterations for an epoch
            if (self.T < self.reset_temp
                or self.num_iter_in_epoch > self.iterations):
                ## the temp is below reset threshold
                self.T = self.init_T
                self.iterations = self.iter_f(self.iterations) # update
                self.num_epoch -= 1 # starting new round of epoch
                self.num_iter_in_epoch = 0

        ## Sample the next scenario
        new_sample = self.proposal_f(self.old_sample, self.num_iter_in_epoch,
                                     self.decay_rate, self.dimension)
        self.last_sample = new_sample

        return new_sample
