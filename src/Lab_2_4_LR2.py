import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import math


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)

    import numpy as np

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (least squares).
        
        Args:
            X (np.ndarray): Independent variable data (2D array) with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Verificar si hay valores NaN o Inf en los datos
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Error: La matriz de entrada contiene valores NaN.")
        if np.isinf(X).any() or np.isinf(y).any():
            raise ValueError("Error: La matriz de entrada contiene valores Inf.")

        # Verificar si hay columnas constantes en X (que causarían singularidad)
        for i in range(X.shape[1]):
            if np.all(X[:, i] == X[0, i]):
                print(f"Advertencia: La columna {i} es constante y puede causar problemas.")

        # Verificar si la matriz (X^T * X) es mal condicionada (potencialmente singular)
        cond_number = np.linalg.cond(X.T @ X)
        if cond_number > 1e10:
            print("Advertencia: La matriz (X^T * X) es mal condicionada. Usando pseudo-inversa.")

        # Aplicar la ecuación de mínimos cuadrados utilizando la pseudo-inversa para evitar singularidad
        W = np.linalg.pinv(X.T @ X) @ X.T @ y

        # Asignar valores a los coeficientes
        self.intercept = W[0]  # Primer coeficiente es la intersección
        self.coefficients = W[1:]  # Resto son los coeficientes del modelo


    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """

        # Número de muestras
        m = len(y)

        # Inicializar los parámetros en valores pequeños cercanos a 0
        self.coefficients = np.random.rand(X.shape[1] - 1) * 0.01  # Coeficientes
        self.intercept = np.random.rand() * 0.01  # Término de intersección

        # Gradiente descendente
        for epoch in range(iterations):
            # Predicciones del modelo
            predictions = X[:, 1:].dot(self.coefficients) + self.intercept

            # Cálculo del error
            error = predictions - y

            # Cálculo del gradiente
            gradient_coeff = (1/m) * X[:, 1:].T.dot(error)  # Gradiente de los coeficientes
            gradient_intercept = (1/m) * np.sum(error)  # Gradiente de la intersección

            # Actualización de parámetros
            self.coefficients -= learning_rate * gradient_coeff
            self.intercept -= learning_rate * gradient_intercept

            # Mostrar el costo cada 1000 iteraciones
            if epoch % 1000 == 0:
                mse = (1/(2*m)) * np.sum(error**2)
                print(f"Epoch {epoch}: MSE = {mse}")

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

        # Paste your code from last week

        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        if np.ndim(X) == 1:
            # TODO: Predict when X is only one variable
            predictions = []
            for i in range(len(X)):
                value = self.intercept + self.coefficients*X[i]
                predictions.append(value)
        
        else:
            # TODO: Predict when X is more than one variable
            predictions = X @ self.coefficients + self.intercept
            

        
        return np.array(predictions)




def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """

    # R^2 Score
    rss = 0
    tss = 0
    for i in range(len(y_true)):
        rss += (y_true[i] - y_pred[i])**2
        tss += (y_true[i]-np.mean(y_true))**2
    r_squared = 1 - (rss/tss)

    # Root Mean Squared Error
    rm = 0
    for i in range(len(y_true)):
        rm += (y_true[i] - y_pred[i])**2
    rmse = math.sqrt(rm/len(y_true))

    # Mean Absolute Error
    m = 0
    for i in range(len(y_true)):
        m += abs(y_true[i] - y_pred[i])
    mae = m/len(y_true)

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


import numpy as np

def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = X.copy()

    for index in sorted(categorical_indices, reverse=True):
        # Extraer la columna categórica
        categorical_column = X_transformed[:, index]

        # Encontrar los valores únicos de la columna (maneja strings)
        unique_values = np.unique(categorical_column)

        # Crear una matriz de codificación one-hot
        one_hot = np.zeros((X_transformed.shape[0], len(unique_values)))

        for i, value in enumerate(unique_values):
            one_hot[:, i] = (categorical_column == value).astype(int)

        # Opción para eliminar la primera categoría y evitar multicolinealidad
        if drop_first:
            one_hot = one_hot[:, 1:]

        # Eliminar la columna original e insertar las nuevas columnas codificadas
        X_transformed = np.delete(X_transformed, index, axis=1)  # Eliminar la columna original
        X_transformed = np.hstack((X_transformed[:, :index], one_hot, X_transformed[:, index:]))  # Insertar nuevas columnas

    return X_transformed
