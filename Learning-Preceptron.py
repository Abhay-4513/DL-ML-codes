##Lerning Perceptron 
import numpy as np  
# Dataset 
# Small = 0 
# Big = 1  
X = np.array([     [1, 1],   # Small     
              [3, 3]    # Big ])  
y = np.array([0, 1])  # Initial values 
weights = np.zeros(X.shape[1]) 
bias = 0 
learning_rate = 1 
epochs = 2  
print("Initial Weights :", weights) 
print("Initial Bias :", bias)  # Training 
for epoch in range(epochs):      
    print("\n===== Epoch", epoch + 1, "=====")      
    for i in range(len(X)):          
      x = X[i]        
      print(x)        
      actual = y[i]         
      print(y)          
      # Step 1: Calculate output         
      output = np.dot(x, weights) + bias         
      print("x",x)         
      print("output",output)         
      # Step 2: Prediction         
      if output >= 0:             
        prediction = 1         
      else:             
        prediction = 0          
      # Step 3: Error         
      error = actual - prediction         
      # Step 4: Learn         
      weights = weights + learning_rate * error * x         
      print("weight=",weights)         
      bias = bias + learning_rate * error         
      print("Bais=",bias)          
      print("\nInput :", x)         
      print("Output :", output)         
      print("Prediction :", prediction)         
      print("Actual :", actual)         
      print("Error :", error)         
      print("Weights :", weights)         
      print("Bias :", bias)  
print("\nTraining Completed") 
print("Final Weights :", weights) 
print("Final Bias :", bias)   
