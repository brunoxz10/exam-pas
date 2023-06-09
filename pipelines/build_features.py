import config
import pandas as pd
import re 


def add_cotas(df: pd.DataFrame) -> pd.DataFrame:
    cotas_columns = [col for col in df.columns if 'classificacao' in col]
    cotas_columns.pop(0) # removing 'classificacao_final_universal'
        
    df['cotista'] = df[cotas_columns].notnull().any(axis=1).astype(int)
    
    for column in cotas_columns:
        colum_name = re.sub("classificacao_final_", "", f'{column}_flag')
        df[colum_name] = df[column].notnull().astype(int)
    
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

def main():
    scores = pd.read_parquet('../data/interim/scores.parquet')
    approvals = pd.read_parquet('../data/interim/approvals.parquet')
    scores = add_cotas(scores)
    df = add_label(scores, approvals)
    df = convert_string_to_float(df, config.NUMERICAL_FEATURES)
    df.to_parquet('../data/processed/scores_approvals.parquet')
    return df


if __name__ == '__main__':
    main()
