import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

class ion_basic:
    def __init__(self, mass, charge):
        self.mass = mass
        self.charge = charge
        self.ion_height = None

def analyze_rf_pot(Y, Z, phi, ion_height_guess):
    Y = (Y.ravel())
    Z = (Z.ravel()-ion_height_guess)
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

    return coeff, np.mean(phi), grad, hessian

def hessian_upper_vector(h):
    h_arr = jnp.array(h, dtype=jnp.float64)
    if h_arr.ndim == 2 and h_arr.shape[0] == h_arr.shape[1]:
        iu = jnp.triu_indices(h_arr.shape[0])
        return h_arr[iu]
    return h_arr.ravel()

def upper_vector_to_hessian(vec):
    size = int((jnp.sqrt(8*len(vec)+1)-1)/2)
    hessian = jnp.zeros((size, size), dtype=jnp.float64)
    iu = jnp.triu_indices(size)
    hessian = hessian.at[iu].set(vec)
    hessian = hessian + hessian.T - jnp.diag(jnp.diag(hessian))
    return hessian

def fit_prep(dc_fields, keys, target_phi, target_grad, target_hessian):
    rows = []

    if target_phi is None:

        for key in keys:
            coeff, phi_mean, grad, hessian = dc_fields[key]
            row = jnp.concatenate([
                jnp.array(grad, dtype=jnp.float64).ravel(),
                hessian_upper_vector(hessian),
            ])
            rows.append(row[None, :])

        A = jnp.vstack(rows)
        b = jnp.concatenate([
            jnp.array(target_grad, dtype=jnp.float64).ravel(),
            hessian_upper_vector(target_hessian),
        ])

    else:
        for key in keys:
            coeff, phi_mean, grad, hessian = dc_fields[key]
            row = jnp.concatenate([
                jnp.array(phi_mean, dtype=jnp.float64).ravel(),
                jnp.array(grad, dtype=jnp.float64).ravel(),
                hessian_upper_vector(hessian),
            ])
            rows.append(row[None, :])

        A = jnp.vstack(rows)
        b = jnp.concatenate([
            jnp.array(target_phi, dtype=jnp.float64).ravel(),
            jnp.array(target_grad, dtype=jnp.float64).ravel(),
            hessian_upper_vector(target_hessian),
        ])

    return A.T, b

# def control_find(A, b):
#     n_inputs = A.shape[1]
#     n_constraints = A.shape[0]

#     # Minimize voltage square within constrain Ax = b
#     kkt_lhs = jnp.block([[jnp.identity(n_inputs), A.T], [A, jnp.zeros((n_constraints, n_constraints))]])
#     kkt_rhs = jnp.concatenate([jnp.zeros(n_inputs), b])
    
#     inputs = jnp.linalg.solve(kkt_lhs, kkt_rhs)[:n_inputs]
#     return inputs

def print_field(b, delta_pos = False):
    if len(b) == 10:
        phi, grad, upper_hessian = b[0], b[1:4], b[4:]
        hessian = upper_vector_to_hessian(upper_hessian)
        print(f"Phi: {phi}")
        print(f"Grad: {grad}")
        print(f"Hessian: {hessian}")
        if delta_pos:
            print(f"Delta Pos (um): {-jnp.linalg.solve(hessian, grad)*1e6}")
        return 
    elif len(b) == 9:
        grad, upper_hessian = b[:3], b[3:]
        hessian = upper_vector_to_hessian(upper_hessian)
        print(f"Grad: {grad}")
        print(f"Hessian: {hessian}")
        if delta_pos:
            print(f"Delta Pos (um): {-jnp.linalg.solve(hessian, grad)*1e6}")