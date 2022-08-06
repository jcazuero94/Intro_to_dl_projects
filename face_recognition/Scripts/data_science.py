import os
import pandas as pd
import numpy as np
import datetime
from tensorflow import keras
import matplotlib.pyplot as plt


CWD = os.getcwd()

def train_test_split(faces_dir:str, faces_vgg: str, train: str, cv: str, test: str):
    """Create train, test and validation split"""
    faces_vgg = pd.read_parquet(CWD + faces_vgg)
    faces_dir = pd.read_parquet(CWD + faces_dir)
    faces_vgg = pd.merge(faces_dir[['gender']],faces_vgg, how='left',left_index=True, right_index=True)
    faces_vgg = faces_vgg[faces_vgg['gender']!='N']
    names = faces_vgg['name'].unique()
    names = np.random.permutation(names)
    train_names = names[:int(0.6*len(names))]
    cv_names = names[int(0.6*len(names)):int(0.8*len(names))]
    test_names = names[int(0.8*len(names)):]
    faces_vgg[faces_vgg['name'].isin(train_names)].drop(['address','number','name'],axis=1).to_parquet(CWD + train)
    faces_vgg[faces_vgg['name'].isin(cv_names)].drop(['address','number','name'],axis=1).to_parquet(CWD + cv)
    faces_vgg[faces_vgg['name'].isin(test_names)].drop(['address','number','name'],axis=1).to_parquet(CWD + test)

def model_training(
    train: str, 
    cv: str,
    model_url: str, 
    loss_fig: str, 
    acc_fig: str
):
    """Train neural network over the preprocesed images to determine gender"""
    # Data prep
    train = pd.read_parquet(CWD + train)
    cv = pd.read_parquet(CWD + cv)
    X_train = train.iloc[:,1:].values
    y_train = train['gender'].apply(lambda x: int(x=='M')).values
    X_cv = cv.iloc[:,1:].values
    y_cv = cv['gender'].apply(lambda x: int(x=='M')).values
    del train, cv
    # Model
    input_faces = keras.Input(shape = (X_train.shape[1]))
    dense_1 = keras.layers.Dense(70, activation='relu')(input_faces)
    dropout_1 = keras.layers.Dropout(0.3)(dense_1)
    dense_2 = keras.layers.Dense(50, activation='relu')(dropout_1)
    dropout_2 = keras.layers.Dropout(0.3)(dense_2)
    dense_3 = keras.layers.Dense(30, activation='relu')(dropout_2)
    predictions = keras.layers.Dense(1, activation='sigmoid')(dense_3)
    model = keras.Model(inputs=input_faces, outputs=predictions)
    opt = keras.optimizers.Adam()
    model.compile(
       opt,
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    # Fit
    loss_train = []
    loss_cv = []
    acc_train = []
    acc_cv = []
    model_best = None
    best_loss = np.inf
    for i in range(30):
        model_fit = model.fit(
            x = X_train,
            y = y_train,
            validation_data=(X_cv, y_cv),
            use_multiprocessing=True,
            verbose=True,
        )
        loss_train += model_fit.history["loss"]
        loss_cv += model_fit.history["val_loss"]
        acc_train += model_fit.history['accuracy']
        acc_cv += model_fit.history['val_accuracy']
        if loss_cv[-1] < best_loss:
            best_loss = loss_cv[-1]
            model_best = keras.models.clone_model(model)
            model_best.build((None, X_train.shape[1]))
            model_best.compile(
                opt,
                loss='binary_crossentropy',
                metrics=['accuracy'],
            )
            model_best.set_weights(model.get_weights())
    model_best.save(CWD + model_url)
    # Training results
    fig_loss, ax = plt.subplots()
    ax.plot(loss_train)
    ax.plot(loss_cv)
    fig_acc, ax = plt.subplots()
    ax.plot(acc_train)
    ax.plot(acc_cv)
    fig_loss.savefig(CWD+loss_fig)
    fig_acc.savefig(CWD+acc_fig)

def main():
    complexity = 2
    faces_vgg = f"/../data/03_primary/faces_vgg_{complexity}"
    faces_dir = "/../data/03_primary/faces_dir_gender"
    train = '/../data/04_model_input/train_gd'
    cv = '/../data/04_model_input/cv_gd'
    test = '/../data/04_model_input/test_gd'
    model_url = '/../data/05_models/model_gd'
    loss_fig = '/../data/06_model_output/train_loss.png'
    acc_fig = '/../data/06_model_output/train_acc.png'
    print(f'Train test split {datetime.datetime.now().strftime("%H:%M:%S")}')
    train_test_split(faces_dir, faces_vgg, train, cv, test)
    print(f'Model training {datetime.datetime.now().strftime("%H:%M:%S")}')
    model_training(train,cv,model_url,loss_fig,acc_fig)

if __name__ == '__main__':
    main()

 