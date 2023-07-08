import config
import pandas as pd
import re 


def add_cotas_flags(df: pd.DataFrame, cotas_columns: list) -> pd.DataFrame:
    """Adds flag column to each type of affirmative quota."""        
    df['cotista'] = df[cotas_columns].notnull().any(axis=1).astype(int)
    
    for column in cotas_columns:
        colum_name = re.sub("classificacao_final_", "", f'{column}_flag')
        df[colum_name] = df[column].notnull().astype(int)
    
    publicas_flags = cotas_columns.copy()
    publicas_flags.remove('classificacao_final_cotas_negros')
    df['publicas_flag'] = df[publicas_flags].notnull().any(axis=1).astype(int)

    return df


def add_label(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Adds the target variable, i.e., approved or not approved."""
    df = pd.merge(df1, df2, how='left', on='numero_inscricao', indicator=True)
    df['label'] = df._merge.apply(lambda x: 1 if x == 'both' else 0)
    return df


def convert_string_to_float(df: pd.DataFrame, colnames: list) -> pd.DataFrame:
    """Converts list of string columns to float."""
    for colname in colnames:
        df[colname] = df[colname].str.replace(' ', "", regex=True)
        df[colname] = df[colname].str.replace('[R$]', "", regex=True)
        df[colname] = df[colname].str.replace(',', ".", regex=False)
        df[colname] = df[colname].apply(float)
    return df


def add_pseudo_argumento_final(df: pd.DataFrame) -> pd.DataFrame:
    """Adds feature that is a weighted average of the main exam components.
    
    Computes the Pseudo Argumento Final (PAF) which is a engineered feature
    that is a proxy for the true Argumento Final.
    """
    df["pseudo_argumento_final"] = (
        df["escore_bruto_p2_etapa1"]
        + 2 * df["escore_bruto_p2_etapa2"]
        + 3 * df["escore_bruto_p2_etapa3"]
    ) / 6

    return df


def get_approved_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Gets statistics of scores of approved students grouped by course.
    
    Computes mean, median, min, max, std statistics of approved students of
    the previous subprogram grouped by course. This will be used to create
    features for the students of the curren subprogram.

    Args:
        df: dataframe of approved students with their scores.

    Returns:
        A pandas DataFrame with summary statistics for each course.
    """
    df_approved = df[df.label == 1]
    approved_stats = df_approved.groupby(["course"], as_index=False).agg(
        {"pseudo_argumento_final": ["mean", "median", "min", "max", "std"]}
    )
    approved_stats.columns = ["course", "mean", "median", "min", "max", "std"]
    approved_stats = approved_stats.sort_values(
        ["median"], ascending=False
    ).reset_index(drop=True)

    return approved_stats


def add_stats_features(
        df: pd.DataFrame,
        df_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Adds statistics flags according to the data of previous subprogram.
    
    Joins the summary statistics of Pseudo Argumento Final of the as dataframe
    that should have the data of the previous subprogram. Adds the final
    engineered features that are flags.

    Args:
        df: data of students of the current subprogram.
        df_stats: summary statistics of PAF of the previous subprogram.

    Returns:
        A pandas dataframe as the final processed data.
    """
    df = pd.merge(df, df_stats, on='course', how='left')
    df['min_flag'] = df['pseudo_argumento_final'] > df['min']
    df['max_flag'] = df['pseudo_argumento_final'] > df['max']
    df['median_flag'] = df['pseudo_argumento_final'] > df['median']
    df['mean_flag'] = df['pseudo_argumento_final'] > df['mean']

    return df


def build_features_wrapper(
    scores_file_path: str,
    approvals_file_path: str,
) -> pd.DataFrame:
    
    scores = pd.read_parquet(scores_file_path)
    approvals = pd.read_parquet(approvals_file_path)
    scores = add_cotas_flags(scores, config.COTAS_COLUMNS)
    df = add_label(scores, approvals)
    df = convert_string_to_float(df, config.NUMERICAL_FEATURES)
    df = add_pseudo_argumento_final(df)

    return df


def main():
    
    # building features for data from subprograma 2019-2021
    scores_file_path = '../../data/interim/scores_2019_2021.parquet'
    approvals_file_path = '../../data/interim/approvals_convocation_2019_2021.parquet'
    
    df = build_features_wrapper(scores_file_path, approvals_file_path)
    approved_stats = get_approved_stats(df)
    approved_stats.to_parquet('../../data/interim/approved_stats_convocation_2019_2021.parquet')
    df = add_stats_features(df, approved_stats)
    df.to_parquet('../../data/processed/scores_approvals_convocation_2019_2021.parquet')

    
    # building features for data from subprograma 2020-2022
    scores_file_path = '../../data/interim/scores_2020_2022.parquet'
    approvals_file_path = '../../data/interim/approvals_convocation_2020_2022.parquet'
    
    df = build_features_wrapper(scores_file_path, approvals_file_path)
    approved_stats = pd.read_parquet('../../data/interim/approved_stats_convocation_2019_2021.parquet')          
    df = add_stats_features(df, approved_stats)
    df.to_parquet('../../data/processed/scores_approvals_convocation_2020_2022.parquet')

    
if __name__ == '__main__':
    main()
