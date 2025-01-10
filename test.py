from fastapi import FastAPI
import uvicorn 
import numpy as np
import random

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
#initialize x as a 5 * 5 matrix
x = [[random.uniform(-1, 1) for _ in range(5)] for _ in range(5) ]

def without_numpy(mat1, mat2):
    """initialized a matrix of 0 to store solutions"""
    solution = [[0 for _ in range(5)] for _ in range(5)]
    """nested loop to iterate through the matrices rows and columns"""
    if len(mat1[0]) != len(mat2):
        raise ValueError("Invalid Matrix Dimensions")
    else:
        for i in range(5):                  #rows of M
            for j in range(5):              #rows of B
                for k in range(5):          #multiplication and addition
                    solution[i][j] += (mat1[i][k] * x[k][j])
                solution[i][j] += mat2[i][j]
    return solution

#Make a call to the function
result = without_numpy(M, B)
#Recreate the function with the sigmoid Function

if __name__ == "__main__":
    uvicorn.run(app)

'''
    Create a requirements.txt
    Upload to render
'''

