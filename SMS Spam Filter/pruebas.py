class PerceptronParalelo:
    def __init__(self, epocas, percep = 10, mu = 1, gamma = 0.05, epsilon = 0.01):
        self.epocas = epocas
        self.tasa_aprendizaje = none
        self.mu = mu                # Importancia de margen limpio
        self.gamma = gamma          # Margen
        self.eps = epsilon      # Tolerancia de error
        self.percep = percep


    def train(self, X_tr, Y_tr):
        # Vector extendido
        ext = np.ones((len(Y_tr), 1))
        self.X_tr = np.append(X_tr, ext, axis=1)

        self.Y_tr = Y_tr
        self.w = np.random.rand(3)  # [-0.5, 0.5, 0]     # Vector de pesos
        self.t = 0  # Cantidad de cambios

        ''' Entrenamiento: Sea pp el producto punto de w con x_i, si
            - Si ô > o + eps and alfa_i * z >= 0, -z
            - Si ô < o - eps and alfa_i * z < 0, -z
            - Si ô <= o + eps and 0 <= alfa_i * z and alfa_i * z < gamma, +z * mu
            - Si ô <= o + eps and 0 <= alfa_i * z and alfa_i * z < gamma, +z * mu
        '''
        for ep in range(1, self.epocas):
            for i in range(len(self.X_tr)):
                pp = np.dot(self.X_tr[i], self.w)
                if pp < 0 and self.Y_tr[i] == 1:  # Suma
                    self.w = self.w + self.X_tr[i]
                    self.t = self.t + 1
                if pp > 0 and self.Y_tr[i] == 0:  # Resta
                    self.w = self.w - self.X_tr[i]
                    self.t = self.t + 1

    def predict(self, X_test):
        # Vector extendido
        ext = np.ones((len(X_test), 1))
        X_test = np.append(X_test, ext, axis=1)

        Y_test = np.zeros(len(X_test))
        for i in range(len(X_test)):
            pp = np.dot(X_test[i], self.w)
            if pp >= 0:
                Y_test[i] = 1
        return Y_test

    def evaluate(self, X_test, Y_test):
        Y_pred = self.predict(X_test)
        exitos = 0
        for i in range(len(Y_test)):
            if Y_test[i] == Y_pred[i]:
                exitos += 1
        return exitos / len(Y_test)

    def peso(self):
        return self.w