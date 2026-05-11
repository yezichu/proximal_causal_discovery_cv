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



def generate_samples_sin(num_samples=1000, type='causal', return_dict=True):
    U = np.random.normal(0, 1, num_samples)
    epsilon_X = np.random.normal(0, 1, num_samples)
    epsilon_W = np.random.normal(0, 1, num_samples)
    epsilon_Z = np.random.normal(0, 1, num_samples)
    epsilon_Y = np.random.normal(0, 1, num_samples)

    # 2. 生成负控制变量 (Negative Controls)
    # W = -2 * sin(U) + epsilon_W
    # Z = 2 * sin(U) + epsilon_Z
    W = -2 * np.sin(U) + epsilon_W
    Z = 2 * np.sin(U) + epsilon_Z

    # 3. 生成治疗变量 (Treatment Assignment)
    # X = 2 * sin(U) + epsilon_X
    X = 2 * np.sin(U) + epsilon_X

    # 4. 生成结果变量 Y
    # 根据 H1 (X 不独立于 Y | U) 或 H0 (X 独立于 Y | U)
    if type == 'causal':
        # Y = X + sin(U) + 2*W^2 + epsilon_Y
        Y = X + np.sin(U) + 2 * (W**2) + epsilon_Y
    else:
        # H0: Y = sin(U) + 2*W^2 + epsilon_Y (即 X 的系数为 0)
        Y = np.sin(U) + 2 * (W**2) + epsilon_Y

    if return_dict:
        return {'X': X, 'U': U, 'W': W, 'Z': Z, 'Y': Y}
    else:
        return X, U, W, Z, Y