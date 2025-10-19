import numpy as np

def generate_samples(gamma_w,num_samples=1000, type='causal', return_dict=True):
    # Gaussian noise
    epsilon_U = np.random.normal(0, 1, num_samples) 
    epsilon_Y = np.random.normal(0, 1, num_samples)
    
    epsilon_X = np.random.normal(0, 1, num_samples)  
    epsilon_W = np.random.normal(0, 1, num_samples) 
    epsilon_Z = np.random.normal(0, 1, num_samples)

    # Structural equations
    U = epsilon_U
    X = 1/np.sqrt(5)*(2 * U + epsilon_X)
    W = 1/np.sqrt(5)*(-2 * U + epsilon_W)
    Z = 1/np.sqrt(5)*(2 * U + epsilon_Z)
    if type == 'causal':
        Y =  1*X + U + gamma_w * W + epsilon_Y
    else:
        Y =  0*X + U + 0.5 * W + epsilon_Y
    if return_dict:
        return {'X': X, 'U': U, 'W': W, 'Z': Z, 'Y': Y}
    else:
        return X, U, W, Z, Y


def generate_samples_nonlinear(gamma_w,num_samples=1000, return_dict=True):
    # Gaussian noise
    epsilon_U = np.random.normal(0, 1, num_samples) 
    epsilon_Y = np.random.normal(0, 1, num_samples)
    
    epsilon_X = np.random.normal(0, 1, num_samples)  
    epsilon_W = np.random.normal(0, 1, num_samples) 
    epsilon_Z = np.random.normal(0, 1, num_samples)

    # Structural equations
    U = epsilon_U
    X =  U + epsilon_X
    W =  U + epsilon_W
    Y =  X**2 + U + gamma_w * W**2 + epsilon_Y
    if return_dict:
        return {'X': X, 'U': U, 'W': W,'Y': Y}
    else:
        return X, U, W, Y
