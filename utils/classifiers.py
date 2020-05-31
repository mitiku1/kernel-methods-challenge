import numpy as np
import cvxopt
from .kernels import linear_kernel, polynomial_kernel, rbf_kernel

def cvxopt_qp(P, q, G, h, A, b):
    P = .5 * (P + P.T)
    cvx_matrices = [
        cvxopt.matrix(M) if M is not None else None for M in [P, q, G, h, A, b] 
    ]
    #cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(*cvx_matrices, options={'show_progress': False})
    return np.array(solution['x']).flatten()
def svm_dual_soft_to_qp_kernel(K, y, C=1):
    n = K.shape[0]
    assert (len(y) == n)
        
    # Dual formulation, soft margin
    # P = np.diag(y) @ K @ np.diag(y)
    P = np.diag(y).dot(K).dot(np.diag(y))
    # As a regularization, we add epsilon * identity to P
    eps = 1e-12
    P += eps * np.eye(n)
    q = - np.ones(n)
    G = np.vstack([-np.eye(n), np.eye(n)])
    h = np.hstack([np.zeros(n), C * np.ones(n)])
    A = y[np.newaxis, :]
    b = np.array([0.])
    return P, q, G, h, A, b
solve_qp = cvxopt_qp




class KernelMethodBase(object):
    '''
    Base class for kernel methods models
    
    Methods
    ----
    fit
    predict
    '''
    kernels_ = {
        'linear': linear_kernel,
        'polynomial': polynomial_kernel,
        'rbf': rbf_kernel
    }
    def __init__(self, **kwargs):
        kernel = kwargs.get("kernel", "rbf")
        self.kernel_function_ = self.kernels_[kernel]
        self.kernel_parameters = self.get_kernel_parameters(**kwargs)
        
    def get_kernel_parameters(self, **kwargs):

        params = {}
        params["kernel"] = kwargs.get('kernel', "rbf")
       
        params['sigma'] = kwargs.get('sigma', 1.)
        params["C"] = kwargs.get("C", 1.0)
        # if self.kernel_name == 'polynomial':
        params['degree'] = kwargs.get('degree', 2)

        return params

    def fit(self, X, y, **kwargs):
        return self
        
    def decision_function(self, X):
        pass

    def predict(self, X):
        pass
    def set_params(self, params):
        self.kernel_parameters.update(params)
    def get_params(self, deep=False):
        return self.kernel_parameters
class KernelSVM(KernelMethodBase):
    '''
    Kernel SVM Classification
    
    Methods
    ----
    fit
    predict
    '''
    def __init__(self, **kwargs):
        
        # Python 3: replace the following line by
        # super().__init__(**kwargs)
        super(KernelSVM, self).__init__(**kwargs)

    def fit(self, X, y, tol=1e-3):
        n, p = X.shape
        assert (n == len(y))
    
        self.X_train = X
        self.y_train = y
        
        # Kernel matrix
        K = self.kernel_function_(
            self.X_train, self.X_train, **self.kernel_parameters)
        
        # Solve dual problem
        self.alpha = solve_qp(*svm_dual_soft_to_qp_kernel(K, y, C=self.kernel_parameters.get("C")))
        
        # Compute support vectors and bias b
        sv = np.logical_and((self.alpha > tol), (self.kernel_parameters.get("C") - self.alpha > tol))
        self.bias = y[sv] - K[sv].dot(self.alpha * y)
        self.bias = self.bias.mean()

        self.support_vector_indices = np.nonzero(sv)[0]

        return self
        
    def decision_function(self, X):
        K_x = self.kernel_function_(X, self.X_train, **self.kernel_parameters)
        return K_x.dot(self.alpha * self.y_train) + self.bias

    def predict(self, X):
        return np.sign(self.decision_function(X))




class KernelSVMMultiK(KernelMethodBase):
    '''
    Kernel SVM Classification
    
    Methods
    ----
    fit
    predict
    '''
    def __init__(self, weights, **kwargs):
        
        # Python 3: replace the following line by
        # super().__init__(**kwargs)
        self.weights = weights
        super(KernelSVMMultiK, self).__init__(**kwargs)

    def fit(self, X, y, tol=1e-3):
        n  = X[0].shape[0]
        assert (n == len(y))
    
        self.X_train = X
        self.y_train = y
        

        K = None
        # Kernel matrix
        for i in range(len(self.X_train)):
            if K is None:
                K = self.kernel_function_(
                        self.X_train[i], self.X_train[i], **self.kernel_parameters) * self.weights[i]
            else:
                K += self.kernel_function_(
                        self.X_train[i], self.X_train[i], **self.kernel_parameters)* self.weights[i]
      
        n = K.shape[0]
        # Solve dual problem
        print(self.kernel_parameters)
        self.alpha = solve_qp(*svm_dual_soft_to_qp_kernel(K, y, C=self.kernel_parameters.get("C")))
        
        # Compute support vectors and bias b
        sv = np.logical_and((self.alpha > tol), (self.kernel_parameters.get("C") - self.alpha > tol))
        self.bias = y[sv] - K[sv].dot(self.alpha * y)
        self.bias = self.bias.mean()

        self.support_vector_indices = np.nonzero(sv)[0]

        return self
        
    def decision_function(self, X):
        # K_x = self.kernel_function_(X, self.X_train, **self.kernel_parameters)

        K_x = None
        # Kernel matrix
        for i in range(len(self.X_train)):
            if K_x is None:
                K_x = self.kernel_function_(
                        X[i], self.X_train[i], **self.kernel_parameters)* self.weights[i]
            else:
                K_x += self.kernel_function_(
                        X[i], self.X_train[i], **self.kernel_parameters)* self.weights[i]
        
        
        
        return K_x.dot(self.alpha * self.y_train) + self.bias

    def predict(self, X):
        return np.sign(self.decision_function(X))