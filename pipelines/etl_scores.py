import config
from utils.scores import *

print(config.RESULTS_PDF) 

text = get_pdf_as_string(config.RESULTS_PDF)
text_cleaned = clean_results(text)
text_split_list = split_results_by_course(text_cleaned)
text_split_list = concatenate_numeros_inscricao(text_split_list)
df_scores = get_results_dataframe(text_split_list)
df_scores = rename_columns(df_scores, config.COLUMN_NAMES)
df_scores = strip_df(df_scores)
df_scores.to_parquet('../data/scores.parquet')

