import numpy as np
from .replay_memory import ReplayMemory
from .SumTree import SumTree

class ReplayMemoryPrior(ReplayMemory):

    def __init__(self, max_size, screen_shape, n_variables, n_features):

        super(ReplayMemoryPrior, self).__init__(max_size, screen_shape, n_variables, n_features)
        self.PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        self.PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        self.PER_b = 0.4  # importance-sampling, from initial value increasing to 1

        self.PER_b_increment_per_sampling = 0.001

        self.absolute_error_upper = 1.

        self.tree = SumTree(max_size)

    def add(self, screen, variables, features, action, reward, is_final):

        #Call Replay Memory Add to change state variables
        super(ReplayMemoryPrior, self).add(screen, variables, features, action, reward, is_final)


        #Update Tree with Priority and index in form of cursor
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        #experience = (screen, variables, features, action, reward, is_final)
        self.tree.add(max_priority, self.cursor-1)  # set the max p for new p


    def get_batch(self, batch_size, hist_size):

        #Add Assertionion  by Checks
        assert self.size > 0, 'replay memory is empty'
        assert hist_size >= 1, 'history is required'

        b_idx, b_ISWeights = np.empty((batch_size,), dtype=np.int32), np.empty((batch_size, 1), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        # print(self.tree.total_priority ,batch_size  )
        priority_segment = self.tree.total_priority / batch_size  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        # p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        #
        # p_min = max(p_min, 0.002)
        # max_weight = (p_min * batch_size) ** (-self.PER_b)
        # max_weight = max( max_weight ,1 )

        idx = np.zeros(batch_size, dtype='int32')
        count = 0
        count2 = 0

        while count < batch_size :
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * count, priority_segment * (count + 1)
            print("Priority Segment", priority_segment)
            print("A = ",a,"B = ",b)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)

            # check that we are not wrapping over the cursor
            #if self.cursor <= data + 1 < self.cursor + hist_size  and data not in [hist_size - 1, self.size - 1]:
             #   continue

            # s_t should not contain any terminal state, so only
            # its last frame (indexed by index) can be final
            #if np.any(self.isfinal[data - (hist_size - 1):data]) and count2 < batch_size:
             #   print("One..............")
              #  count2 += 1
               # continue

            #count2 = 0
            #priorSum = self.tree.getSum(index, hist_size)
            # P(j)
            sampling_probabilities = priority / (self.tree.total_priority )

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[count, 0] = np.power(batch_size * sampling_probabilities, -self.PER_b)
            #print('cpo ',count,b_ISWeights[count, 0],max_weight,np.power(batch_size * sampling_probabilities, -self.PER_b) )
            b_idx[count] = index

            idx[count] = data
            count +=1

        all_indices = idx.reshape((-1, 1)) + np.arange(-(hist_size - 1), 2)
        screens = self.screens[all_indices]
        variables = self.variables[all_indices] if self.n_variables else None
        features = self.features[all_indices] if self.n_features else None
        actions = self.actions[all_indices[:, :-1]]
        rewards = self.rewards[all_indices[:, :-1]]
        isfinal = self.isfinal[all_indices[:, :-1]]

        # check batch sizes
        assert idx.shape == (batch_size,)
        assert screens.shape == (batch_size, hist_size + 1) + self.screen_shape
        assert (variables is None or variables.shape == (batch_size,
                                                         hist_size + 1, self.n_variables))
        assert (features is None or features.shape == (batch_size,
                                                       hist_size + 1, self.n_features))
        assert actions.shape == (batch_size, hist_size)
        assert rewards.shape == (batch_size, hist_size)
        assert isfinal.shape == (batch_size, hist_size)

        return dict(
            screens=screens,
            variables=variables,
            features=features,
            actions=actions,
            rewards=rewards,
            isfinal=isfinal,
            tree_weights=b_ISWeights,
            tree_index=b_idx
        )

    def batch_update(self,tree_idx, abs_errors):
        # print(abs_errors)
        abs_errors= abs_errors.data.numpy()
        # print(abs_errors)

        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        #print(tree_idx)
        # print('pss',ps)
        for ti, p in zip(list(tree_idx), list(ps)):
            self.tree.update(ti, p)