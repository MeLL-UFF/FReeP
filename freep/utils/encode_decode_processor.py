from sklearn.preprocessing import LabelEncoder
from numbers import Number
import pandas as pd
import numpy as np


def encode(X, y):
    if not isinstance(y.values[0], Number):
        y_encoder = LabelEncoder()
        y = pd.Series(y_encoder.fit_transform(y.values))
    else:
        y_encoder = None
    X = pd.get_dummies(X, prefix_sep="#")
    return X, y, y_encoder

def encode_preference(
    preferences, encoded_preferences, current_preferences, preferences_encoder
):
    resp = []
    for current_preference in current_preferences:
        # a não ser que já seja uma preferencia numerica
        if not isinstance(preferences[current_preference][0], Number):
            # verifica se essa feature codificada esta entre as preferencias
            preference = current_preference + "#" + preferences[current_preference][0]
            if preference in preferences_encoder:
                if encoded_preferences[preference][0] == 1:
                    resp.append(preference)
            else:
                resp.append(current_preference)
        else:
            resp.append(current_preference)
    return resp


def decode_y(y, y_encoder):
    if y_encoder is None:
        return y
    else:
        return y_encoder.inverse_transform(y)
