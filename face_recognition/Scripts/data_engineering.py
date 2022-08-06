import os
import pandas as pd
import datetime
from tensorflow import keras
from keras.utils import img_to_array, load_img
from keras.applications.vgg16 import preprocess_input, VGG16


CWD = os.getcwd()

def create_faces_dir(faces_url: str, faces_dir: str):
    """Creates dir of labelled images of faces"""
    faces = os.listdir(CWD + faces_url)
    faces_df = pd.DataFrame(columns=["address"], data=faces)
    faces_df["number"] = faces_df["address"].apply(
        lambda x: int(x.split("_")[-1].split(".")[0])
    )
    faces_df["name"] = faces_df["address"].apply(lambda x: " ".join(x.split("_")[:-1]))
    faces_df.to_parquet(CWD +faces_dir)

def vgg_feature_extraction(faces_url, faces_dir, faces_vgg, complexity):
    """Extract features using VGG model"""
    faces_dir_df = pd.read_parquet(CWD + faces_dir)
    model = VGG16()
    model = keras.Model(inputs=model.inputs, outputs=model.layers[-complexity].output)
    def predict_vgg(im_url, model=model, pref=faces_url):
        image = load_img(CWD + pref +'/'+ im_url, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, *image.shape))
        image = preprocess_input(image)
        return model.predict(image, verbose=False)[0]
    faces_dir_df["vgg_features"] = faces_dir_df["address"].apply(predict_vgg) 
    len_vgg = len(faces_dir_df.iloc[0]["vgg_features"])
    faces_dir_df[[f"vgg_{i}" for i in range(len_vgg)]] = [
        list(x) for x in faces_dir_df["vgg_features"].values
    ]
    faces_dir_df.drop("vgg_features", axis=1, inplace=True)
    faces_dir_df.to_parquet(CWD + faces_vgg,)
    
def gender_assign(faces_dir, faces_dir_gender):
    """Assign gender to dataset according to name"""
    from names import names_f, names_m
    faces_dir = pd.read_parquet(CWD + faces_dir)
    faces_dir['gender'] = faces_dir['name'].apply(
        lambda x: 'M' if sum([n in x.split(' ') for n in names_m]) > 0 else ('F' if sum([n in x.split(' ') for n in names_f]) > 0 else 'N') 
    )
    faces_dir[faces_dir['gender'] != 'N'].to_parquet(CWD + faces_dir_gender)

def main():
    faces_url = '/../data/01_raw/Faces'
    faces_dir = '/../data/02_intermediate/faces_dir'
    faces_dir_gender = '/../data/03_primary/faces_dir_gender'
    complexity = 2
    faces_vgg = f"/../data/03_primary/faces_vgg_{complexity}"
    # print(f'Create faces directory {datetime.datetime.now().strftime("%H:%M:%S")}')
    # create_faces_dir(faces_url, faces_dir)
    # print(f'VGG feature extraction {datetime.datetime.now().strftime("%H:%M:%S")}')
    # vgg_feature_extraction(faces_url, faces_dir, faces_vgg, complexity)
    print(f'Gender Assign {datetime.datetime.now().strftime("%H:%M:%S")}')
    gender_assign(faces_dir,faces_dir_gender)

if __name__ == '__main__':
    main()

 