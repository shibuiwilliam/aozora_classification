# coding: utf-8


"""
This is data integrating and training script for author classification using Keras with character-level cnn.
Be sure to get the text files from Aozora-bunko and convert them to UTF-8 encoded csv files, just by running aozora-scrape.py.
"""


import sys, os.path, re, csv, os, glob
import pandas as pds
import numpy as np
import zipfile
import codecs
from keras.layers import Activation, Dense, Dropout, Flatten, Convolution2D, MaxPooling2D, Reshape, Input, merge
from keras.models import Model, Sequential
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, Callback, CSVLogger, ModelCheckpoint

# Setting work directory and opening csv files.
base_url = "http://www.aozora.gr.jp/"
data_dir = "./"
aozora_dir = data_dir + "aozora_data/"
log_dir = aozora_dir + "log/"
target_author_file = data_dir + "target_author.csv"

auth_target = []
with open(target_author_file,"r") as f:
    reader = csv.reader(f)
    for row in reader:
        auth_target.append(row)
auth_target



# Generating integrated file of csv's of all the pieces for each author.
# One csv file is generated and saved for each author.

def author_data_integ(auth_target=auth_target):
    for w in auth_target[1:]:
        print ("starting: " + w[0])
        auth_dir = '{}{}/'.format(aozora_dir, w[0])
        csv_dir = '{}{}'.format(auth_dir, "csv/")
        files = os.listdir(csv_dir)
        integ_np = np.array([["author", "line"]])
        for file in files:
            if "csv" in file:
                print ("   now at: " + file)
                file_name = csv_dir + file
                pds_data = pds.read_csv(file_name, index_col=0)
                pds_data = pds_data.dropna()
                np_data = np.array(pds_data.ix[:,[0,2]])

                out = [j for j in range(len(np_data)) if '-----------' in str(np_data[j,1])]
                if not out: out = [1]
                hyphen_pos = int(out[len(out) - 1])

                last_20 = len(np_data) - 20

                np_data = np_data[hyphen_pos+1:last_20,:]
                integ_np = np.vstack((integ_np, np_data))

        integ_pds = pds.DataFrame(integ_np[1:,:], columns=integ_np[0,:])
        integ_pds.to_csv(auth_dir + w[0] + '_integ.csv', quoting=csv.QUOTE_ALL)
        print ("finished: " + w[0])


author_data_integ()

# Loading integrated csv files.

def load_integ(author, auth_dir):
    integ_csv = auth_dir + author + "_integ.csv"
    data = pds.read_csv(integ_csv)
    return data



natsume_data = load_integ(auth_target[1][0], auth_dir = '{}{}/'.format(aozora_dir, auth_target[1][0]))
np_natsume = np.array(natsume_data.ix[1:,1:])

akutagawa_data = load_integ(auth_target[2][0], auth_dir = '{}{}/'.format(aozora_dir, auth_target[2][0]))
np_akutagawa = np.array(akutagawa_data.ix[1:,1:])

mori_data = load_integ(auth_target[3][0], auth_dir = '{}{}/'.format(aozora_dir, auth_target[3][0]))
np_mori = np.array(mori_data.ix[1:,1:])

sakaguchi_data = load_integ(auth_target[4][0], auth_dir = '{}{}/'.format(aozora_dir, auth_target[4][0]))
np_sakaguchi = np.array(sakaguchi_data.ix[1:,1:])

yoshikawa_data = load_integ(auth_target[5][0], auth_dir = '{}{}/'.format(aozora_dir, auth_target[5][0]))
np_yoshikawa = np.array(yoshikawa_data.ix[1:,1:])



# Preparing arrays

natsume_txt = np.array([np_natsume[:,1]]).T
akutagawa_txt = np.array([np_akutagawa[:,1]]).T
mori_txt = np.array([np_mori[:,1]]).T
sakaguchi_txt = np.array([np_sakaguchi[:,1]]).T
yoshikawa_txt = np.array([np_yoshikawa[:,1]]).T




natsume_id = np.array([np.zeros(len(np_natsume))]).T
akutagawa_id = np.array([np.zeros(len(np_akutagawa)) + 1]).T
mori_id = np.array([np.zeros(len(np_mori)) + 2]).T
sakaguchi_id = np.array([np.zeros(len(np_sakaguchi)) + 3]).T
yoshikawa_id = np.array([np.zeros(len(np_yoshikawa)) + 4]).T




natsume = np.hstack((natsume_txt, natsume_id))
akutagawa = np.hstack((akutagawa_txt, akutagawa_id))
mori = np.hstack((mori_txt, mori_id))
sakaguchi = np.hstack((sakaguchi_txt, sakaguchi_id))
yoshikawa = np.hstack((yoshikawa_txt, yoshikawa_id))




print (len(natsume), len(akutagawa), len(mori), len(sakaguchi), len(yoshikawa))




"""
This time, I am using pieces from Soseki Natsume, Ryunosuke Akutagawa, Ogai Mori and Ango Sakaguchi
for they have about the same number of lines.
Omitting Eiji Yoshikawa due to the fact that he has three time more line of texts than others.
"""

data_integ = np.vstack((np.vstack((np.vstack((natsume, akutagawa)),mori)),sakaguchi))



# Converting each line of texts to encoded array.

def load_data(txt, max_length=200):
    txt_list = []
    for l in txt:
        txt_line = [ord(x) for x in str(l).strip()]
        # You will get encoded text in array, just like this
        # [25991, 31456, 12391, 12399, 12394, 12367, 12387, 12390, 23383, 24341, 12391, 12354, 12427, 12290]
        txt_line = txt_line[:max_length]
        txt_len = len(txt_line)
        if txt_len < max_length:
            txt_line += ([0] * (max_length - txt_len))
        txt_list.append((txt_line))
    return txt_list




# Making arrays for training text and target author
txt_list = load_data(txt = data_integ[:,0])
np_txt = np.array(txt_list)

tgt_list = data_integ[:,1]
np_tgt_list = np_utils.to_categorical(tgt_list)



# Creating character-level convolutional neural network model.

def create_model(embed_size=128, max_length=200, filter_sizes=(2, 3, 4, 5), filter_num=64):
    inp = Input(shape=(max_length,))
    emb = Embedding(0xffff, embed_size)(inp)
    emb_ex = Reshape((max_length, embed_size, 1))(emb)
    convs = []
    for filter_size in filter_sizes:
        conv = Convolution2D(filter_num, filter_size, embed_size, activation="relu")(emb_ex)
        pool = MaxPooling2D(pool_size=(max_length - filter_size + 1, 1))(conv)
        convs.append(pool)
    convs_merged = merge(convs, mode='concat')
    reshape = Reshape((filter_num * len(filter_sizes),))(convs_merged)
    fc1 = Dense(64, activation="relu")(reshape)
    bn1 = BatchNormalization()(fc1)
    do1 = Dropout(0.5)(bn1)
    fc2 = Dense(4, activation='sigmoid')(do1)
    model = Model(input=inp, output=fc2)
    return model



# Training the model.

def train(inputs, targets, batch_size=100, epoch_count=100, max_length=200, model_filepath=aozora_dir + "model.h5", learning_rate=0.001):
    start = learning_rate
    stop = learning_rate * 0.01
    learning_rates = np.linspace(start, stop, epoch_count)

    model = create_model(max_length=max_length)
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    
    
    target = os.path.join('/tmp', 'weights.*.hdf5')
    files = [(f, os.path.getmtime(f)) for f in glob.glob(target)]
    if len(files) != 0:
        latest_saved_model = sorted(files, key=lambda files: files[1])[-1]
        model.load_weights(latest_saved_model[0])
    
    # Logging file for each epoch
    csv_logger_file = '/tmp/clcnn_training.log'
    
    # Checkpoint model for each epoch
    checkpoint_filepath = "/tmp/weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5"

    model.fit(inputs, targets,
              nb_epoch=epoch_count,
              batch_size=batch_size,
              verbose=1,
              validation_split=0.1,
              shuffle=True,
              callbacks=[
                  LearningRateScheduler(lambda epoch: learning_rates[epoch]),
                  CSVLogger(csv_logger_file),
                  ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, save_best_only=True, save_weights_only=False, monitor='val_acc')
              ])

    model.save(model_filepath)




if __name__ == "__main__":
    train(np_txt, np_tgt_list)



