import numpy as np

class ion_basic:
    def __init__(self, mass, charge):
        self.mass = mass
        self.charge = charge
        self.ion_height = None

def analyze_rf_pot(Y, Z, phi, ion_height_guess):
    Y = Y.ravel()
    Z = Z.ravel()-ion_height_guess
    phi = phi.ravel()

    A = np.column_stack([
    Y**2,
    Z**2,
    Y,
    Z,
    np.ones_like(Y)
    ])

    coeff, *_ = np.linalg.lstsq(A, phi, rcond=None)
    a, b, c, d, e = coeff
    ion_height = -d/(2*b)+ion_height_guess

    grad = np.array([0, 0, 0])
    hessian = np.array([[0, 0, 0], [0, 2*a, 0], [0, 0, 2*b]])
    
    return coeff, grad, hessian, ion_height

def analyze_dc_pot(X, Y, Z, phi, x, y, z):
    X = X.ravel()
    x0 = np.mean(x)
    X = X - x0
    x = x - x0

    Y = Y.ravel()
    y0 = np.mean(y)
    Y = Y - y0
    y = y - y0
   
    Z = Z.ravel()
    z0 = np.mean(z)
    Z = Z - z0
    z = z - z0

    phi = phi.ravel()

    A = np.column_stack([
    X**2 - Z**2,
    Y**2 - Z**2,
    X*Y,
    X*Z,
    Y*Z,
    X,
    Y,
    Z,
    np.ones_like(Y)
    ])

    coeff, *_ = np.linalg.lstsq(A, phi, rcond=None)
    a, b, d, e, f, g, h, i, j = coeff
    c = -a - b
    grad = np.array([2*a*x + d*y + e*z + g, 2*b*y + d*x + f*z + h, 2*c*z + e*x + f*y + i])
    hessian = np.array([[2*a, d, e], [d, 2*b, f], [e, f, 2*c]])
    
    return coeff, grad, hessian
