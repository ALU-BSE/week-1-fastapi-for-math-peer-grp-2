from fastapi import FastAPI, Request
import uvicorn 
import numpy as np

app = FastAPI()

def generate_random_matrix():
    return np.random.randint(-10, 10, size=(5, 5))  # Random integers between 0 and 9

# Initialize M and B as np arrays
M = generate_random_matrix();
B = generate_random_matrix();
# use the post decorator directly below this
'''
    Initialize M and B as np arrays
'''
#Implement the formula MX + B
#Have two function one using numpy and another not using numpy
#Return 
@app.post("/calculate")
async def f(request: Request):
    body = await request.json()

    matrix = body.get("matrix")

    if not matrix or len(matrix) != 5 or any(len(row) != 5 for row in matrix):
        return {"error": "Matrix must be 5x5"}


    X = np.array(matrix)

    # Calculate using numpy
    numpy_result = with_numpy(M, X, B)
    
    # Calculate without numpy
    non_numpy_result = without_numpy(M, X, B)

    # Apply sigmoid function
    sigmoid_result = sigmoid(numpy_result)
    
    return {
        "matrix_multiplication": numpy_result.tolist(),
        "non_numpy_multiplication": non_numpy_result,
        "sigmoid_output": sigmoid_result.tolist()
    }

# Function to calculate the matrix multiplication using numpy
def with_numpy(M, X, B):
    return np.matmul(M, X) + B

# Function to calculate the matrix multiplication without using numpy
def without_numpy(mat1, X, mat2):
    """initialized a matrix of 0 to store solutions"""
    solution = [[0 for _ in range(5)] for _ in range(5)]
    """nested loop to iterate through the matrices rows and columns"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        raise ValueError("Invalid Matrix Dimensions")
    else:
        for i in range(5):                  #rows of M
            for j in range(5):              #rows of B
                for k in range(5):          #multiplication and addition
                    solution[i][j] += int(mat1[i][k] * X[k][j])
                solution[i][j] += int(mat2[i][j])
    return solution

# Function to apply the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

'''
    Create a requirements.txt
    Upload to render
'''

