from itertools import combinations

words = ['shape', 'color', 'material', 'function', 'size']

all_combinations = []

for r in range(1, len(words) + 1):
    combos = combinations(words, r)
    all_combinations.extend(combos)

# print all combination
for combo in all_combinations:
    print(combo)
    
print(f'length is {len(all_combinations)}')

