import numpy as np

# utility fxn, determines if prediction is a 1 or 0
def threshold(y_pred):
    if y_pred >= 0.5:
        return 1.0
    else:
        return 0.0

# Load data
trainingData = np.loadtxt("https://raw.githubusercontent.com/michaelchapa/" \
                          "ai_machineLearning_assign3/master/gd-train.dat", skiprows = 1)
testData = np.loadtxt("https://raw.githubusercontent.com/michaelchapa/" \
                      "ai_machineLearning_assign3/master/gd-test.dat", skiprows = 1)

rows, attributes = trainingData.shape # attributes = columns
rows2, attributes2 = testData.shape

# Get all Learning Rates, from 0.05 to 1 (inclusive)
alphas = []
alpha = 0.05
while alpha <= 1.05:
    alphas.append(alpha)
    alpha += 0.05

# Learn for each learning rate, print results, do it again
for alpha in alphas:
    weights = np.zeros(attributes - 1) # weights initialized to 0
    
    # Train the weights
    for instance in trainingData: # each row in the data
        y_pred = np.dot(weights, instance[0:-1]) # make y prediction
        error = y_pred - instance[-1]
        
        # update each weight
        for j in range(0, 13): 
            weights[j] = weights[j] - alpha * error * instance[j]
    
    # Check accuracy on trainingData
    correctCount = 0
    for instance in trainingData:
        y_pred = np.dot(weights, instance[0:-1]) # predict
        y_pred = threshold(y_pred)
        
        if y_pred == instance[-1]:
            correctCount += 1
        
    accuracy = (correctCount / rows) * 100.0
    print("Accuracy for LR of %.2lf on Training set = %.0lf" \
          % (alpha, accuracy), end = "%\n")
    
    # Check accuracy on testData
    correctCount = 0
    for instance in testData:
        y_pred = np.dot(weights, instance[0:-1])
        y_pred = threshold(y_pred)
        
        if y_pred == instance[-1]:
            correctCount += 1
            
    accuracy = (correctCount / rows2) * 100.0
    print("Accuracy for LR of %.2lf on Testing set =  %.0lf" \
          % (alpha, accuracy), end = "%\n")
        
    print()

        
    

        
    
    
    