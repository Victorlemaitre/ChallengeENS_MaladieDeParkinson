import polars as pl
import torch
from torch.utils.data import TensorDataset


def polars_data_pipeline(train_input, train_label, config):
    """
    What this pipeline does :
        Numerizes categorical features
        Normalizes columns, replace null values by the mean and create for each column a "flag" 0-1 column which indicates the presence of a null values
        Concatenates all the rows with the same patient id and pad them to a fixed length
    Inputs and outputs are polars dataframes
    """

    #One hot encoding of the 'gene' and 'cohort' columns
    train_input = train_input.to_dummies(columns=['gene'])

    dict_cohort = {
        'A': 0,
        'B': 1,
    }
    train_input = train_input.with_columns(pl.col('cohort').replace(dict_cohort).cast(pl.Int64).alias('cohort'))


    #Add a new feature 'time since diagnosis'
    train_input = train_input.with_columns((pl.col('age') - pl.col('age_at_diagnosis')).alias('time_since_diagnosis'))


    #Concatenation of the rows with the same patient id
    train_label = train_label.drop('Index')
    input_label = pl.concat([train_input, train_label], how='horizontal')
    df2 = input_label.group_by('patient_id').agg(
        [pl.col(col) for col in input_label.columns if col not in ['Index', 'patient_id']])

    df3 = df2.with_columns(
        [

            pl.col(col).list.get(i, null_on_oob=True).alias(col + f"{i}")
            for i in range(config.context_length)
            for col in df2.columns if col != 'patient_id'
        ]
    )

    target_columns = ['target' + f'{i}' for i in range(config.context_length)]
    train_label = df3.select(target_columns)
    train_input = df3.drop([col for col in df2.columns]).drop(target_columns)

    #Normalization of columns
    for j in range(config.context_length):
        cols_to_normalize_values = [
            'time_since_intake_on' + f'{j}',
            'time_since_intake_off' + f'{j}',
            'ledd' + f'{j}',
            'age_at_diagnosis' + f'{j}',
            'time_since_diagnosis' + f'{j}',
            'age' + f'{j}',
            'on' + f'{j}',
            'off' + f'{j}'
        ]

        train_input = train_input.with_columns([
            ((pl.col(col) - pl.col(col).mean()) / (pl.col(col).std() + 1e-8))
            .alias(col)
            for col in cols_to_normalize_values
        ])

    #Filling of the missing values + creation of flag column with a 1 if the corresponding entry is null
    nb_feature_before = train_input.shape[1] // config.context_length
    for j in range(config.context_length):

        cols_missing_values = [
            'time_since_intake_on' + f'{j}',
            'time_since_intake_off' + f'{j}',
            'ledd' + f'{j}',
            'age_at_diagnosis' + f'{j}',
            'time_since_diagnosis' + f'{j}',
            'on' + f'{j}',
            'off' + f'{j}'

        ]

        train_input = train_input.with_columns([
            pl.when(pl.col(col).is_null())
            .then(0)
            .otherwise(1)
            .alias('flag_' + col)
            for col in cols_missing_values
        ])

        #Re-order the columns so that each flag is in the right #nb_feature box
        columns = train_input.columns
        for i, col in enumerate(cols_missing_values):
            columns.remove('flag_' + col)
            columns.insert(j * (nb_feature_before + len(cols_missing_values)) + i, 'flag_' + col)
        train_input = train_input.select(columns)

    train_label = train_label.fill_null(config.null_value)
    train_input = train_input.fill_null(config.null_value)

    return train_input, train_label


def data_pipeline(file_features, file_labels, config):
    data_features = pl.read_csv(file_features)
    data_labels = pl.read_csv(file_labels)

    data_train_inputs, data_train_labels = polars_data_pipeline(data_features, data_labels, config)

    X = torch.tensor(data_train_inputs.to_numpy(), dtype=torch.float32)
    y = torch.tensor(data_train_labels.to_numpy(), dtype=torch.float32)

    return TensorDataset(X, y), X, y

