import numpy as np

SENSOR_ALPHA2 = 0.001070
SENSOR_EPSILON2 = 0.000998
S_POOLED = 0.0114


def build_sensor_covariance(n):
    covar = np.ones((n, n)) * SENSOR_ALPHA2
    np.fill_diagonal(covar, np.diag(covar) + SENSOR_EPSILON2)
    return covar


def build_sentinel_covariance_n(alpha2, epsilon2, n):
    off_diagonal = alpha2 + SENSOR_ALPHA2
    diagonal = off_diagonal + epsilon2 + SENSOR_EPSILON2
    covar = np.full((n, n), off_diagonal)
    np.fill_diagonal(covar, diagonal)
    return covar


def build_sample_covariance(obs_stdev_list):
    return np.diag(np.array(obs_stdev_list) ** 2)


def build_combined_covariance(n_observations, n_obs_continuous, sensor_type='sensor',
                              obs_stdev_list=None, sentinel_alpha2=None,
                              sentinel_epsilon2=None):
    if sensor_type == 'sensor':
        Sigma_obs = build_sensor_covariance(n_obs_continuous)
    elif sensor_type == 'sentinel':
        if sentinel_alpha2 is None or sentinel_epsilon2 is None:
            raise ValueError(
                "sentinel_alpha2 and sentinel_epsilon2 required for sentinel sensor_type")
        Sigma_obs = build_sentinel_covariance_n(sentinel_alpha2, sentinel_epsilon2,
                                                 n_obs_continuous)
    else:
        raise ValueError(f"Unknown sensor_type: {sensor_type}")

    n_samples = n_observations - n_obs_continuous
    if n_samples > 0 and obs_stdev_list is not None:
        if len(obs_stdev_list) != n_samples:
            raise ValueError(
                f"obs_stdev_list length {len(obs_stdev_list)} != n_samples {n_samples}")
        Sigma_samp = build_sample_covariance(obs_stdev_list)
        Sigma = np.zeros((n_observations, n_observations))
        Sigma[:n_obs_continuous, :n_obs_continuous] = Sigma_obs
        Sigma[n_obs_continuous:, n_obs_continuous:] = Sigma_samp
    else:
        Sigma = Sigma_obs
    return Sigma