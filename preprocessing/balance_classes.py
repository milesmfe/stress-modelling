from imblearn.under_sampling import RandomUnderSampler


def underSample(X, y):
    rus = RandomUnderSampler(sampling_strategy='not minority')
    X_res, y_res = rus.fit_resample(X, y)
    return X_res, y_res