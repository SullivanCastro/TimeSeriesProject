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
    
    relative_errors = (y_true - y_pred) / y_true * (y_true != 0)
    relative_errors = relative_errors[~np.isnan(relative_errors)]
    
    return np.mean(relative_errors**2,)

def rmsre(y_true, y_pred):
    """
    Calcule le Root Mean Squared Relative Error (RMSRE)
    Args:
        y_true: valeurs réelles (array-like)
        y_pred: valeurs prédites (array-like)
    Returns:
        Le RMSRE
    """
    # complete with nan if not the same length
    if len(y_true) < len(y_pred):
        y_true = np.append(y_true, [np.nan] * (len(y_pred) - len(y_true)))
    elif len(y_pred) < len(y_true):
        y_pred = np.append(y_pred, [np.nan] * (len(y_true) - len(y_pred)))

    idx = np.where(~np.isnan(y_true) & ~np.isnan(y_pred))

    msre_ = msre(y_true[idx], y_pred[idx])
    rmsre = np.sqrt(msre_)
    return rmsre