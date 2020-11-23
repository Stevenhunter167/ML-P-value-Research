class Weighted_avg_ensemble:

    def __init__(self, classifiers=[Distance_classifier, PCA_pvalue], device='cpu'):
        # init classifiers
        self.classifiers = classifiers
        # init parameters
        self.device = device

    # @staticmethod
    # def unit_projection(t):
    #     # return None
    #     with torch.no_grad():
    #         t.set_(t / torch.norm(t))

    def fit_submodels(self, Xtrain: np.array, Ytrain: np.array):
        # record meta data
        self.classes = np.array(np.unique(Ytrain), dtype=np.long)

        # fit all submodels in the ensemble
        Ytrain = np.array(Ytrain, dtype=np.long)
        self.models = []
        for Classifier in self.classifiers:
            model = Classifier(Xtrain, Ytrain)
            model.fit()
            self.models.append(model)

        # init parameters
        self.parameters = torch.rand((len(self.classifiers), len(self.classes)),
                                     dtype=torch.float64, requires_grad=True, device=self.device)

    def forward(self, X: np.array):
        predictions = []
        for classifier in self.models:
            predictions.append(classifier.all_p_values(X))
        P = torch.tensor(predictions).to(self.device).permute(1,0,2)
        r = torch.sum(torch.mul(P, self.parameters**2), axis=1)
        phat = torch.mul(r, 1/torch.sum(self.parameters**2, axis=0))
        return phat

    def all_p_values(self, X):
        with torch.no_grad():
            return self.forward(X)

    def acc(self, Xtest: np.array, Ytest: np.array):
        Ytest = torch.tensor(Ytest, dtype=torch.float64, device=self.device)

        correct = torch.sum(Ytest==torch.argmax(self.all_p_values(Xtest), axis=1))
        total   = Ytest.shape[0]
        return correct / total