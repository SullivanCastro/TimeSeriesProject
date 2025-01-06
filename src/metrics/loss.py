import numpy as np

def msre(y_true, y_pred):
    """
    Calcule le Mean Squared Relative Error (MSRE)
    Args:
        y_true: valeurs réelles (array-like)
        y_pred: valeurs prédites (array-like)
    Returns:
        MSRE
    """
    relative_errors = (y_true - y_pred) / y_true
    return np.mean(relative_errors**2)

def rmsre(y_true, y_pred):
    """
    Calcule le Root Mean Squared Relative Error (RMSRE)
    Args:
        y_true: valeurs réelles (array-like)
        y_pred: valeurs prédites (array-like)
    Returns:
        Le RMSRE
    """
    idx = np.where(~np.isnan(y_true) & ~np.isnan(y_pred))

    msre_ = msre(y_true[idx], y_pred[idx])
    rmsre = np.sqrt(msre_)
    return rmsre