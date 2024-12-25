import pickle
import pandas as pd


with open('data.pickle', 'rb') as file:
    dataset = pickle.load(file)


data = dataset.get('data', [])
labels = dataset.get('labels', [])


df = pd.DataFrame(data)
df['Label'] = labels  


print(df.head(1000))



df.style.set_table_attributes("style='display:inline'").set_caption("Dataset Overview")


"""import pickle
import pandas as pd


with open('data.pickle', 'rb') as file:
    dataset = pickle.load(file)


labels = dataset.get('labels', [])


labels_df = pd.DataFrame(labels, columns=['Label'])


print(labels_df.head(1000))  """
