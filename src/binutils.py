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