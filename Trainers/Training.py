class Training:

    def train(self,mlp, X, T, X_val, T_val, n_epochs=1000, eps=1e-6, threshold=0.5, suppress_print=False):
        raise NotImplementedError("Metodo astratto")