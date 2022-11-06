
import copy

class KnapsackSolver:

    def __init__(self, N, weight_arr, profit_arr, max_allowable_weight) -> None:
        self.N = N
        self.weight_arr = weight_arr
        self.profit_arr = profit_arr
        assert N == len(self.weight_arr) == len(self.profit_arr)

        self.max_allowable_weight = max_allowable_weight
        self.dp_max_weight = None
        self.dp_state_tracker = None

    def solve(self):

        self.dp_max_weight = [[0] * (self.max_allowable_weight + 1) for i in range(self.N)]
        self.dp_state_tracker = [[None] * (self.max_allowable_weight + 1) for i in range(self.N)]

        for i in range(0, self.N):

            curr_weight = self.weight_arr[i]
            curr_profit = self.profit_arr[i]

            for j in range(0, self.max_allowable_weight + 1):
                if j == 0:
                    self.dp_max_weight[i][j] = 0
                    continue

                if i == 0:
                    # if first element, set it to this eleement
                    if curr_weight <= j:
                        self.dp_max_weight[i][j] = curr_profit

                        temp_arr = copy.deepcopy(self.dp_state_tracker[i][j])
                        if temp_arr is None:
                            temp_arr = [i]
                        else:
                            temp_arr.append(i)
                        self.dp_state_tracker[i][j] = temp_arr
                    continue

                # logic on whether we should use "curr_node" only or "curr_node" plus "previous_node" or "previous node" only

                if j - curr_weight < 0:
                    self.dp_max_weight[i][j] = self.dp_max_weight[i-1][j]
                    self.dp_state_tracker[i][j] = copy.deepcopy(self.dp_state_tracker[i-1][j])
                    continue

                # dp_max_weight[i-1][j] ---> JUst use the previous weight
                # dp_max_weight[i-1][j-curr_weight] + curr_profit --> Can add the current weight as well!

                if self.dp_max_weight[i-1][j] > self.dp_max_weight[i-1][j-curr_weight] + curr_profit:
                    self.dp_state_tracker[i][j] = copy.deepcopy(self.dp_state_tracker[i-1][j])
                    self.dp_max_weight[i][j] = self.dp_max_weight[i-1][j]
                else:
                    temp_arr = copy.deepcopy(self.dp_state_tracker[i-1][j-curr_weight])
                    if temp_arr is None:
                        temp_arr = [i]
                    else:
                        temp_arr.append(i)
                    # temp_arr.append(i)
                    self.dp_state_tracker[i][j] = temp_arr

                    self.dp_max_weight[i][j] = self.dp_max_weight[i-1][j-curr_weight] + curr_profit

                self.dp_max_weight[i][j] = max(self.dp_max_weight[i-1][j], self.dp_max_weight[i-1][j-curr_weight] + curr_profit)
