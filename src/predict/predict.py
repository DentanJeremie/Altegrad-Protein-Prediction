import typing as t

import numpy as np
import pandas as pd

from src.utils.constants import FEATURE_WEIGHTS
from src.utils.pathtools import project
from src.utils.logging import logger

def main():
    """Does the final prediction as a weighted mean of other predictions.
    """
    features_dfs: t.List[pd.DataFrame] = list()
    weights_list: t.List[float] = list()

    for feature in FEATURE_WEIGHTS:
        logger.debug(f'Loading predictions of {feature}')
        features_dfs.append(
            pd.read_csv(
                project.get_latest_features(feature),
            )
        )
        weights_list.append(FEATURE_WEIGHTS[feature])

    # Initializing the weighted mean
    logger.info('Computing the weighted mean of the predictions...')
    numeric_columns = list(item for item in features_dfs[0].columns if 'class' in item)
    weighted_mean = np.zeros_like(features_dfs[0][numeric_columns].values)

    for df, weight in zip(features_dfs, weights_list):
        weighted_mean += weight * df[numeric_columns].values

    name_column = features_dfs[0]['name'].values
    output_numpy = np.column_stack((name_column, weighted_mean))
    output_pandas = pd.DataFrame(output_numpy, columns=['name'] + numeric_columns)

    # Storing the output
    output_path = project.get_new_submission_file()
    output_pandas.to_csv(
        output_path,
        index=False,
    )

    logger.info(f'Final prediction stored at {project.as_relative(output_path)}')

if __name__ == '__main__':
    main()
