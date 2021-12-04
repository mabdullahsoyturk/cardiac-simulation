results1, results2 = [], []

with open("../results/test_single.txt") as result_single:
    lines = result_single.readlines()

    for line in lines:
        left, right = line.split(":")
        results1.append((left, right))

with open("../results/test_version3.txt") as result_version1:
    lines = result_version1.readlines()

    for line in lines:
        left, right = line.split(":")
        results2.append((left, right))

for result1, result2 in zip(results1, results2):
    if result1[1] != result2[1]:
        print("Different!!!")
        print(f'Result1 {result1[0]}:{result1[1]}Result2 {result2[0]}:{result2[1]}')
        break
