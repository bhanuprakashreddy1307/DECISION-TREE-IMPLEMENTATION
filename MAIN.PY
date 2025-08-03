import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# Simulated dataset: [Age, Salary, Owns House]
X = [
    [22, 35000, 0], [25, 40000, 0], [47, 85000, 1], [52, 90000, 1], [46, 80000, 1],
    [56, 100000, 1], [32, 60000, 1], [34, 58000, 0], [29, 52000, 0], [40, 75000, 1],
    [30, 49000, 0], [28, 50000, 0], [33, 59000, 1], [36, 61000, 1], [27, 46000, 0],
    [45, 78000, 1], [50, 91000, 1], [26, 45000, 0], [24, 43000, 0], [23, 39000, 0],
    [42, 79000, 1], [44, 77000, 1], [31, 57000, 1], [35, 60000, 1], [48, 82000, 1],
    [55, 96000, 1], [38, 71000, 1], [39, 72000, 1], [21, 36000, 0], [22, 37000, 0],
    [41, 74000, 1], [43, 76000, 1], [49, 83000, 1], [37, 67000, 1], [36, 68000, 1],
    [53, 93000, 1], [54, 95000, 1], [20, 34000, 0], [29, 53000, 0], [33, 56000, 1],
    [34, 55000, 0], [30, 51000, 0], [27, 47000, 0], [25, 41000, 0], [23, 38000, 0],
    [50, 92000, 1], [51, 89000, 1], [46, 81000, 1], [35, 64000, 1], [31, 59000, 1],
    [28, 52000, 0], [26, 46000, 0], [21, 33000, 0], [45, 80000, 1], [38, 70000, 1],
    [40, 76000, 1], [39, 73000, 1], [24, 44000, 0], [32, 65000, 1], [44, 77000, 1]
]

# Target: Buy Product (1 = Yes, 0 = No)
y = [
    0, 0, 1, 1, 1,
    1, 1, 0, 0, 1,
    0, 0, 1, 1, 0,
    1, 1, 0, 0, 0,
    1, 1, 1, 1, 1,
    1, 1, 1, 0, 0,
    1, 1, 1, 1, 1,
    1, 1, 0, 0, 1,
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    0, 0, 0, 1, 1,
    1, 1, 0, 1, 1
]

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree classifier
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Visualize the decision tree
plt.figure(figsize=(16, 10))
plot_tree(clf,
          filled=True,
          feature_names=["Age", "Salary", "Owns_House"],
          class_names=["No", "Yes"])
plt.title("Decision Tree - Buy Product Classification")
plt.show()
