import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (mean=0, variance=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = models.Sequential([
    layers.Input(shape=(4,)),                 # 4 features in Iris dataset
    layers.Dense(16, activation='relu'),      # First hidden layer
    layers.Dense(8, activation='relu'),       # Second hidden layer
    layers.Dense(3, activation='softmax')     # Output layer (3 classes)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train the model
history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=8, 
    validation_split=0.1, 
    verbose=1
)

# Evaluate performance on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\n--- Evaluation Results ---")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")