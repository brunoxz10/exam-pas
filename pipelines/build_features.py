import config
import pandas as pd
import re 


def add_cotas_flags(df: pd.DataFrame, cotas_columns: list) -> pd.DataFrame:
        
    df['cotista'] = df[cotas_columns].notnull().any(axis=1).astype(int)
    
    for column in cotas_columns:
        colum_name = re.sub("classificacao_final_", "", f'{column}_flag')
        df[colum_name] = df[column].notnull().astype(int)
    
    publicas_flags = cotas_columns
    publicas_flags.remove('classificacao_final_cotas_negros')
    df['publicas_flag'] = df[publicas_flags].notnull().any(axis=1).astype(int)

    return df


def add_label(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(df1, df2, how='left', on='numero_inscricao', indicator=True)
    df['label'] = df._merge.apply(lambda x: 1 if x == 'both' else 0)
    return df


def convert_string_to_float(df, colnames):
    for colname in colnames:
        df[colname] = df[colname].str.replace(' ', "", regex=True)
        df[colname] = df[colname].str.replace('[R$]', "", regex=True)
        df[colname] = df[colname].str.replace(',', ".", regex=False)
        df[colname] = df[colname].apply(float)
    return df


def add_pseudo_argumento_final(df):
    df["pseudo_argumento_final"] = (
        df["escore_bruto_p2_etapa1"]
        + 2 * df["escore_bruto_p2_etapa2"]
        + 3 * df["escore_bruto_p2_etapa3"]
    ) / 6

    return df


def get_approved_stats(df: pd.DataFrame):
    df_approved = df[df.label == 1]
    approved_stats = df_approved.groupby(["course"], as_index=False).agg(
        {"pseudo_argumento_final": ["mean", "median", "min", "max", "std"]}
    )
    approved_stats.columns = ["course", "mean", "median", "min", "max", "std"]
    approved_stats = approved_stats.sort_values(
        ["median"], ascending=False
    ).reset_index(drop=True)

    return approved_stats


def add_stats_features(df: pd.DataFrame, df_stats: pd.DataFrame) -> pd.DataFrame:
    
    df = pd.merge(df, df_stats, on='course', how='left')
    df['dist_min'] = df['pseudo_argumento_final'] > df['min']
    df['dist_max'] = df['pseudo_argumento_final'] > df['max']
    df['dist_median'] = df['pseudo_argumento_final'] > df['median']
    df['dist_mean'] = df['pseudo_argumento_final'] > df['mean']

    return df


def main():
    
    scores_file_path = '../data/interim/scores_2020_2022.parquet'
    approvals_file_path = '../data/interim/approvals_2020_2022_complete.parquet'
    
    scores = pd.read_parquet(scores_file_path)
    approvals = pd.read_parquet(approvals_file_path)
    scores = add_cotas_flags(scores, config.COTAS_COLUMNS)
    df = add_label(scores, approvals)
    df = convert_string_to_float(df, config.NUMERICAL_FEATURES)
    df = add_pseudo_argumento_final(df)

    if ['2019_2021' in path for path in [scores_file_path, approvals_file_path]]:
        approved_stats = get_approved_stats(df)
        approved_stats.to_parquet('../data/interim/approved_stats_2019_2021.parquet')
            
    elif ['2020_2022' in path for path in [scores_file_path, approvals_file_path]]:
        approved_stats = pd.read_parquet('../data/interim/approved_stats_2019_2021.parquet')
            
    df = add_stats_features(df, approved_stats)
    df.to_parquet('../data/processed/scores_approvals_2020_2022.parquet')
    
    return df


if __name__ == '__main__':
    main()
