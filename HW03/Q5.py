import matplotlib.pyplot as plt

# Table
instances = [
    (0.95, '+'),
    (0.85, '+'),
    (0.8, '-'),
    (0.7, '+'),
    (0.55, '+'),
    (0.45, '-'),
    (0.4, '+'),
    (0.3, '+'),
    (0.2, '-'),
    (0.1, '-')
]

num_neg = sum(1 for _, label in instances if label == '-')
num_pos = sum(1 for _, label in instances if label == '+')

TP = 0
FP = 0
last_TP = 0
roc_points = [(0,0)]  # Starting point

for i in range(1, len(instances) + 1):
    c_i, y_i = instances[i-1]

    if i > 1:
        c_prev, y_prev = instances[i-2]
        if c_i != c_prev and y_i == '-' and TP > last_TP:
            FPR = FP / num_neg
            TPR = TP / num_pos
            roc_points.append((FPR, TPR))
            last_TP = TP
    
    if y_i == '+':
        TP += 1
    else:
        FP += 1

FPR = FP / num_neg
TPR = TP / num_pos
roc_points.append((FPR, TPR))

# Plotting 
FPR_values, TPR_values = zip(*roc_points)  
print("FPR",FPR_values)
print("TPR",TPR_values)
plt.figure()
plt.plot(FPR_values, TPR_values, marker='o', linestyle='-', color='b',linewidth=3)
plt.plot([0, 1], [0, 1], linestyle='--', color='green')  
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.grid(True)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()
