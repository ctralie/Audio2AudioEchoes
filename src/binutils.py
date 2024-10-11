import numpy as np

def str2binary(s):
    """
    Convert a string to an ASCII binary representation

    Parameters
    ----------
    s: string
        An ASCII string of length N
    
    Returns
    -------
    ndarray(N*8)
        Array of 1's and 0's
    """
    b = ""
    for c in s:
        c = bin(ord(c))[2::]
        c = "0"*(8-len(c)) + c
        b += c
    return np.array([int(c) for c in b])

def binary2str(b):
    """
    Convert an ASCII binary representation back to a

    Parameters
    ----------
    ndarray(N*8) or str
        Array of 1's and 0's, or a string of 0's and 1's

    Returns
    -------
    s: string
        An ASCII string of length N
    """
    b = [[0, 1][c] for c in b]
    b = np.array(b)
    b = np.reshape(b, (len(b)//8, 8))
    b = np.sum( np.array([[2**(i-1) for i in range(8, 0, -1)]]) * b, axis=1)
    return "".join([chr(x) for x in b])

def text2binimg(s, N):
    """
    Create a binary logo out of a string of a specified resolution

    Parameters
    ----------
    s: string
        String to convert
    N: int
        Resolution of image
    
    Returns
    -------
    ndarray(N, N)
        Rasterized image of 1's and 0's
    """
    import matplotlib.pyplot as plt
    lines = s.split("\n")
    W = max([len(s) for s in lines])
    H = len(lines)*1.2
    sz = int(np.floor(N/max(W, H)))
    font = {'family': 'serif',
            'weight': 'normal',
            'size': sz}
    fig = plt.figure(figsize=(N/100, N/100))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, W)
    ax.set_ylim(-H, 0)
    for i, s in enumerate(lines):
        ax.text(0, -1.2-i*1.2, s, fontdict=font)
    ax.axis("off")
    fig.canvas.draw()
    # https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = np.array(data)[:, :, 0]
    data[data < 255] = 0
    data = data/255
    return data


def fix_pn_2(p):
    """
    Map a pseudorandom sequence onto the space of sequences of runs
    of length more than 2, as per [1]

    [1] Yong Xiang, Dezhong Peng, Iynkaran Natgunanathan, Wanlei Zhou
    "Effective Pseudonoise Sequence and Decoding Function for Imperceptibility 
    and Robustness Enhancement in Time-Spread Echo-Based Audio Watermarking"

    Parameters
    ----------
    p: ndarray(L)
        Pseudorandom sequence in {-1, 1}^L

    Returns
    -------
    q: ndarray(L)
        Pseudorandom sequence with no runs of length more than 2
    """
    L = p.size
    q = np.zeros(L)
    q[0] = p[0]
    q[-1] = p[-1]
    for n in range(1, L-1):
        y = (q[n-1] + p[n-1] + p[n] + p[n+1])
        if y > 0:
            y = y//4
        else:
            y = -((-y)//4)
        q[n] = ((-1)**y)*p[n]
    return q

def get_hadamard_basis(L):
    """
    Compute the Hadamard basis for size L

    Returns
    -------
    ndarray(log2(L), L)
        The Hadamard basis
    """
    k = int(np.log2(L))
    B = np.zeros((k, L), dtype=int)
    for i in range(k):
        v = 0
        for j in range(L):
            if j % (L/(2**(i+1))) == 0:
                v = (v+1)%2
            B[i, j] = v
    return B

def get_hadamard_codes(L):
    """
    Compute the (non-augmented) hadamard codes for size L

    Returns
    -------
    ndarray(L, L, dtype=int)
        Hadamard codes
    """
    B = get_hadamard_basis(L)
    X = np.zeros((L, L), dtype=int)
    for i in range(L):
        b = [int(x) for x in bin(i)[2:]]
        b = [0]*(B.shape[0]-len(b)) + b
        b = np.array(b)
        X[i, :] = (b.dot(B)) % 2
    return X