from pydoc import Helper
import pandas as pd
from helperFunctions import get_epochs
from helperFunctions import truncate_df_with_interval

start_date = str(input("Enter start time (yyyy-mm-dd hh:mm:ss): "))
end_date = str(input("Enter end time (yyyy-mm-dd hh:mm:ss): "))

epoch_start, epoch_end = get_epochs(start_date, end_date)
 