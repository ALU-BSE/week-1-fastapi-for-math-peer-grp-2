from fastapi import FastAPI
import uvicorn 
import numpy as np 
import matplotlib.pyplot as plt 



app = FastAPI()

# use the post decorator directly below this
'''
    Initialize M and B as np arrays
'''
M = np.array([[1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5]])

B =np.array([[1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5]])

   
#Implement the formula MX + B
#Have two function one using numpy and another not using numpy
#Return 
@app.post("/")
def f(x):
    return np.matmul(M, x) + B
 
def without_numpy(x):
    return "" 

#initialize x as a 5 * 5 matrix
X = np.ones((5, 5))
#Make a call to the function
f(X)
#Recreate the function with the sigmoid Function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

Y = f(X)
Z = sigmoid(Y)
print(Y)
print(Z.tolist())

if __name__ == "__main__":
    uvicorn.run(app)

'''
    Create a requirements.txt
    Upload to render
'''

