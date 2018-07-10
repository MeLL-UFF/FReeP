from sklearn.preprocessing import LabelEncoder
from numbers import Number
import pandas as pd
import numpy as np


class EncodingProcessor():

    def encode(self, X, y, preferences):
        self.columns_names = list(X)
        if not isinstance(y.values[0], Number):
            self.y_encoder = LabelEncoder()
            y = pd.Series(self.y_encoder.fit_transform(y.values))
        else:
            self.y_encoder = None
        X = pd.get_dummies(X)
        self.X_encoder = list(X)
        preferences = pd.get_dummies(preferences)
        self.preferences_encoder = list(preferences)
        # no_set_columns = list(set(self.X_encoder) - set(self.preferences_encoder))
        # for no_set_column in no_set_columns:
        #     preferences[no_set_column] = np.nan
        return X, y, preferences

    def encode_preference(self, preferences, encoded_preferences, current_preferences):
        resp = []
        for current_preference in current_preferences:
            #a não ser que já seja uma preferencia numerica
            if not isinstance(preferences[current_preference][0], Number):
                #verifica se essa feature codificada esta entre as preferencias
                preference = current_preference + '_' + \
                    preferences[current_preference][0]
                if preference in self.preferences_encoder:
                    if encoded_preferences[preference][0] == 1:
                        resp.append(preference)
                else:
                    resp.append(current_preference)
            else:
                resp.append(current_preference)
        return resp

    def decode_y(self, y):
        if self.y_encoder is None:
            return y
        else:
            return self.y_encoder.inverse_transform(y)
