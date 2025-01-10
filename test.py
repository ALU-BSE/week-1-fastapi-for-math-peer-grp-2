from fastapi import FastAPI
import uvicorn 
import numpy as np 

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
    """nested loop to iterate through the matrices rows and columns"""
    for i in range(5):                  #rows of M
        for j in range(5):              #rows of B
            for k in range(5):          #multiplication and addition
                solution[i][j] += (M[i][k] * B[k][j])
    return solution

#initialize x as a 5 * 5 matrix
solution = [[0 for _ in range(5)] for _ in range(5)]

#Make a call to the function
without_numpy(solution)
#Recreate the function with the sigmoid Function

if __name__ == "__main__":
    uvicorn.run(app)

'''
    Create a requirements.txt
    Upload to render
'''

