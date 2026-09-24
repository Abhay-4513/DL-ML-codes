import numpy as np
import matplotlib.pyplot as plt
def unitStep(v):
    if v >= 0:
        return 1
    else:
        return 0

def perceptronModel(x, w, b):
    v = np.dot(w, x) + b
    y = unitStep(v)
    return y

# OR Logic Function
# w1 = 1, w2 = 1, b = -0.5
def OR_logicFunction(x):
    w = np.array([1, 1])
    b = -0.5
    return perceptronModel(x, w, b)

# testing the Perceptron Model
test1 = np.array([0, 1])
test2 = np.array([1, 1])
test3 = np.array([0, 0])
test4 = np.array([1, 0])

print("OR({}, {}) = {}".format(0, 1, OR_logicFunction(test1)))
print("OR({}, {}) = {}".format(1, 1, OR_logicFunction(test2)))
print("OR({}, {}) = {}".format(0, 0, OR_logicFunction(test3)))
print("OR({}, {}) = {}".format(1, 0, OR_logicFunction(test4)))

inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Get perceptron outputs
outputs = np.array([
    OR_logicFunction(x) for x in inputs
])

# Create plot
plt.figure(figsize=(7, 6))

# Plot Output = 0
plt.scatter(
    inputs[outputs == 0, 0],
    inputs[outputs == 0, 1],
    s=150,
    marker='o',
    label='Output = 0'
)

# Plot Output = 1
plt.scatter(
    inputs[outputs == 1, 0],
    inputs[outputs == 1, 1],
    s=150,
    marker='x',
    label='Output = 1'
)

# Decision boundary
x1 = np.linspace(-0.5, 1.5, 100)
x2 = 0.5 - x1

plt.plot(
    x1,
    x2,
    label='Decision Boundary'
)

# Add point labels
for x, y in zip(inputs, outputs):
    plt.text(
        x[0] + 0.03,
        x[1] + 0.03,
        f'({x[0]}, {x[1]}) → {y}'
    )

# Axis labels
plt.xlabel("Input x₁")
plt.ylabel("Input x₂")

# Title
plt.title("OR Gate using Perceptron")

# Axis limits
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)

# Tick values
plt.xticks([0, 1])
plt.yticks([0, 1])

# Grid
plt.grid(True)

# Legend
plt.legend()

# Display graph
plt.show()