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


def main():
    
    scores = pd.read_parquet('../data/interim/scores_2019_2021.parquet')
    approvals = pd.read_parquet('../data/interim/approvals_2019_2021_complete.parquet')
    scores = add_cotas_flags(scores, config.COTAS_COLUMNS)
    df = add_label(scores, approvals)
    df = convert_string_to_float(df, config.NUMERICAL_FEATURES)
    df = add_pseudo_argumento_final(df)
    df.to_parquet('../data/processed/scores_approvals_2019_2021_complete.parquet')
    
    return df


if __name__ == '__main__':
    main()
