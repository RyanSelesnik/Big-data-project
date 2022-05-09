import csv
from datetime import datetime
# Get quartile assuming the input array is sorted


def getQuartile(arr, quartile):
    n = len(arr)
    if n % 2 == 0:
        return (arr[int(n*quartile) - 1] + arr[int(n*quartile)])/2
    else:
        return arr[int(n*quartile)]


def get_epochs(start_date, end_date):
    epoch_start = datetime.strptime(
        start_date, "%Y-%m-%d %H:%M:%S").timestamp()
    epoch_end = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").timestamp()
    epoch_start = '%.0f' % epoch_start
    epoch_start = int(str((epoch_start.ljust(19, "0"))))
    epoch_end = '%.0f' % epoch_end
    epoch_end = int(str((epoch_end.ljust(19, "0"))))
    return epoch_start, epoch_end


def truncate_df_with_interval(df, epoch_start, epoch_end):
    df.drop(df[df['time'] < epoch_start].index, inplace=True)
    df.drop(df[df['time'] > epoch_end].index, inplace=True)


def write_to_csv(data, output_file_name):
    fieldnames = ['file_name', 'lower_quartile', 'median', 'upper_quartile',
                  'interquartile_range', 'minimum', 'maximum', 'lower_fence', 'upper_fence']

    rows = [data]
    with open('statistical_indicators.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def preprocess_data(df):
    num_null = df.isna().any(axis=1).sum()
    if num_null/len(df) > 0.05:
        raise ValueError(
            "There are too many many missing values, find another data set")
    else:
        df.dropna(inplace=True)
