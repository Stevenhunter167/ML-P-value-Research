import numpy as np
import scipy

class PCA_pvalue:
    def __init__(self, Xtrain, Ytrain):
        """ X in Rn, Y in range(0,n) """
        self.features = Xtrain.shape[1]
        self.classes = np.array(np.unique(Ytrain), dtype='int')
        
        self.Ws  = []
        self.VTs = []
        self.invVTs = []
        
        for yi in self.classes:
            Wi, VTi = self.svd_WVT(Xtrain, Ytrain, yi)
            self.Ws.append(Wi)
#             plt.scatter(Wi[:,0], Wi[:,1])
#             plt.axis('equal')
#             plt.show()
            self.VTs.append(VTi)
            self.invVTs.append(np.linalg.pinv(VTi))

    def fit(self):
        pass
    
    @staticmethod
    def svd_WVT(X, Y, y):
        """ 
        X = U Sigma VT 
        W = U Sigma
        return W , VT
        """
        U,Sigma,VT = scipy.linalg.svd(X[Y==y])
        t = np.concatenate([Sigma, np.zeros((U.shape[0]-Sigma.shape[0],))], axis=0)
        Sc = np.diag(t)
        W = U.dot(Sc)
        VTc = np.concatenate([VT, np.zeros((U.shape[1]-VT.shape[0], VT.shape[1]))], axis=0)
        return W, VTc
    
    @staticmethod
    def empirical_pvalue(X, test):
        """
            X: r by c
            test: 1 by c
            return: 1 by c
        """
        rawP = np.sum(X < test, axis=0) / X.shape[0]
        rawP[rawP>0.5] = 1-rawP[rawP>0.5]
        return rawP
    
    @staticmethod
    def gaussian_pvalue(X, test):
        mean = np.mean(X, axis=0)
        std  = np.std(X, axis=0)
        z = (test - mean) / std
        pvalue = scipy.stats.norm.cdf(z)
        pvalue[pvalue > 0.5] = 1 - pvalue[pvalue > 0.5]
        return pvalue
        
    
    def pvalue(self, Xtest, distribution=None, verbose=0):
        if distribution == None:
            distribution = self.gaussian_pvalue
        

        pv_yi = []
        
        for y in self.classes:
            # W, VT^-1
            W = self.Ws[y]
            
            invVT = self.invVTs[y]
            Wtest = Xtest.dot(invVT)[:, :self.features]
            
            p_test = lambda data: distribution(W[:,:self.features], data)
            pv_yi.append( np.apply_along_axis(arr=Wtest, axis=1, func1d=p_test) )
        
        return np.transpose(np.array(pv_yi), axes=(1,0,2))
    
#     def pvalue_avg(self, Xtest):
#         ''' taking the average of all component '''
#         return np.mean(self.pvalue(Xtest), axis=2)
    
    def pvalue_min(self, Xtest, distribution=None):

        ''' taking the min of all component (suggested by prof. Hayes) '''
        
        if distribution == None:
            distribution = self.gaussian_pvalue
        
        return np.min(self.pvalue(Xtest, distribution=distribution), axis=2)

    def all_p_values(self, Xtest):
        return self.pvalue_min(Xtest) * 2
    
    def predict(self, Xtest, distribution=None):
        
        if distribution == None:
            distribution = self.gaussian_pvalue
        
        return np.array(np.argmax(self.pvalue_min(Xtest, distribution=distribution), axis=1), dtype='int16')
    
    def evaluate_performance(self, Xtest, Ytest, distribution=None):
        
        if distribution == None:
            distribution = self.gaussian_pvalue
        
        return np.sum(self.predict(Xtest, distribution=distribution)==Ytest) / Xtest.shape[0]