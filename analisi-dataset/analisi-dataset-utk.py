import numpy as np
import pandas as pd
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

dataset_folder_name = 'Dataset/utkface/image'
dataset_dict = {
    'razza_id': {
        0: 'bianchi',
        1: 'neri',
        2: 'asiatici',
        3: 'indiani',
        4: 'altri'
    },
    'sesso_id': {
        0: 'Maschi',
        1: 'Femmine'
    }
}

def parse_dataset(dataset_path, ext='jpg'):
    def parse_info_from_file(path):
        try:
            filename = os.path.split(path)[1]
            filename = os.path.splitext(filename)[0]
            eta, sesso, razza, _ = filename.split('_')

            return int(eta), dataset_dict['sesso_id'][int(sesso)], dataset_dict['razza_id'][int(razza)]
        except Exception as ex:
            return None, None, None

    files = glob.glob(os.path.join(dataset_path, "*.%s" % ext))

    records = []
    for file in files:
        info = parse_info_from_file(file)
        records.append(info)

    df = pd.DataFrame(records)
    df['file'] = files
    df.columns = ['Età', 'Sesso', 'Razza', 'file']
    df = df.dropna()
    return df

dataset_dict['sesso_alias'] = dict((g, i) for i, g in dataset_dict['sesso_id'].items())
dataset_dict['razza_alias'] = dict((r, i) for i, r in dataset_dict['razza_id'].items())

df = parse_dataset('Dataset/utkface/image')
df.head()
print(df)

def plot_distribution(pd_series):
    labels = pd_series.value_counts().index.tolist()
    counts = pd_series.value_counts().values.tolist()
    pie_plot = go.Pie(labels=labels, values=counts, hole=.3)
    fig = go.Figure(data=[pie_plot])
    fig.update_layout(title_text='Distribuzione per %s' % pd_series.name)
    fig.show()

#plot_distribution(df['Sesso'])
#plot_distribution(df['Razza'])

coppie = [0, 10, 20, 30, 40, 60, 80, np.inf]
nomi = ['<10', '10-20', '20-30', '30-40', '40-60', '60-80', '80+']
intervalli_eta = pd.cut(df['Età'], coppie, labels=nomi)

plot_distribution(intervalli_eta)
