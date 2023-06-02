import config
from utils.scores import get_pdf_as_string, rename_columns
from utils.labels import *

print(config.APPROVALS_PDF) 

text = get_pdf_as_string(config.APPROVALS_PDF)
text_cleaned = delete_page_titles(text)
text_split_list = split_results_by_student(text_cleaned)
text_split_list_students = filtering_only_students(text_split_list)
df_approvals = get_approvals_dataframe(text_split_list_students)
df_approvals = rename_columns(df_approvals, config.APPROVALS_COLUMNS_NAMES)
df_approvals.to_parquet('../data/processed/approvals.parquet')