results1 = []
results2 = []

iter_num = 0

with open("test_single.txt") as result_single:
    lines = result_single.readlines()

    for line in lines:
        left, right = line.split(":")
        if left != "Iteration":
            results1.append((left, right, iter_num))
        else:
            iter_num = int(right)

with open("test_version5.txt") as result_version1:
    lines = result_version1.readlines()
    
    for line in lines:
        left, right = line.split(":")
        if left != "Iteration":
            results2.append((left, right, iter_num))
        else:
            iter_num = int(right.strip())

for result1, result2 in zip(results1, results2):
    if result1[1] != result2[1]:
        print("Different!!!")
        print(f'Result1 {result1[0]}:{result1[1]}Result2 {result2[0]}:{result2[1]}')
        break
        

