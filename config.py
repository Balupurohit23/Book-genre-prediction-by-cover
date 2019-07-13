train_var = ['loss', 'accuracy']
val_var = ['val_loss', 'val_accuracy']

path = "./data/book_data/"

# width | height
# input_dims = (224, 224)
input_dims = (165, 224)
epochs = 100
batch_size = 16


train_data_path = './data/train_small.json'
val_data_path = './data/val_small.json'

splitter = 30

# Classes are kept sorted