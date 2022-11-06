N = 3
profit_arr = [60, 100, 120] # [3, 2, 6]
weight_arr = [10, 20, 30] # [6, 8, 15]
MAX_ALLOWABLE_WEIGHT = 50 # 6

# val = [60, 100, 120]
# wt = [10, 20, 30]
# W = 50

dp_max_weight = [[0] * (MAX_ALLOWABLE_WEIGHT + 1) for i in range(N)]


for i in range(0, N):

    curr_weight = weight_arr[i]
    curr_profit = profit_arr[i]

    for j in range(0, MAX_ALLOWABLE_WEIGHT + 1):
        if j == 0:
            dp_max_weight[i][j] = 0
            continue

        if i == 0:
            # if first element, set it to this eleement
            if curr_weight <= j:
                dp_max_weight[i][j] = curr_profit
            continue

        # logic on whether we should use "curr_node" only or "curr_node" plus "previous_node" or "previous node" only

        if j - curr_weight < 0:
            dp_max_weight[i][j] = dp_max_weight[i-1][j]
            continue

        # dp_max_weight[i-1][j] ---> JUst use the previous weight
        # dp_max_weight[i-1][j-curr_weight] + curr_profit --> Can add the current weight as well!
        dp_max_weight[i][j] = max(dp_max_weight[i-1][j], dp_max_weight[i-1][j-curr_weight] + curr_profit)

    # print(dp_max_weight)

for arr in dp_max_weight:
    print(arr)

