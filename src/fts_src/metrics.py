import numpy as np

def mean_squared_relative_error(y_true, y_pred):
    """
    Calcule le Mean Squared Relative Error (MSRE)
    Args:
        y_true: valeurs réelles (array-like)
        y_pred: valeurs prédites (array-like)
    Returns:
        Le MSRE
    """
    relative_errors = (y_true - y_pred) / y_true
    msre = np.mean(relative_errors**2)
    return msre

def root_mean_squared_relative_error(y_true, y_pred):
    """
    Calcule le Root Mean Squared Relative Error (RMSRE)
    Args:
        y_true: valeurs réelles (array-like)
        y_pred: valeurs prédites (array-like)
    Returns:
        Le RMSRE
    """
    msre = mean_squared_relative_error(y_true, y_pred)
    rmsre = np.sqrt(msre)
    return rmsre