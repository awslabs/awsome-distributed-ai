import pandas as pd
import torch
from sklearn import preprocessing
import numpy as np
import yaml
import ast
import os
import glob

def get_channel_name_combi(channel_combi_num, channel_dict):
    name_of_channel_combi = ""
    for channel_number in iter(str(channel_combi_num)):
        name_of_channel_combi = "_".join([name_of_channel_combi, channel_dict[int(channel_number)]])
    return name_of_channel_combi

def get_channel_number_combi(channel_names, channel_dict):
    channel_combi = ""
    for channel_name in channel_names.split('_'):
        for key, value in channel_dict.items():
            if value == channel_name:
                channel_combi = "".join([channel_combi, str(key)])
    return channel_combi

def get_channel_name_combi_list(selected_channels, channel_dict):
    channel_names = []
    for channel_combi in selected_channels:
        channel_names.append(get_channel_name_combi(channel_combi,channel_dict))
    return channel_names

config_file = '../scDINO_full_pipeline.yaml'
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

name_of_run = config['meta']['name_of_run']
sk_save_dir = config['meta']['output_dir']
save_dir_downstream_run = sk_save_dir+"/"+name_of_run


#channel_aTub_model_checkpoint2_features.csv
features_path = f"{save_dir_downstream_run}/CLS_features/"

features_file = glob.glob(features_path+'*_features.csv')[0]

labels_file = f"{save_dir_downstream_run}/CLS_features/class_labels.csv"


dino_vit_name = config['train_scDINO']['dino_vit_name']

selected_channels = [config['meta']['selected_channel_combination_per_run']]
selected_channels = list(ast.literal_eval(selected_channels[0]))

channel_dict = config['meta']['channel_dict']
channel_dict = dict(zip(selected_channels, channel_dict.split(',')))
channel_names=get_channel_name_combi_list(selected_channels, channel_dict)

save_dir = f"{save_dir_downstream_run}/{dino_vit_name}_channel{channel_names[0]}_analyses/"

###iterate through all runs and save the results in one file
##for run_index, features in enumerate(snakemake.input["features"]):
    #data

features = np.genfromtxt(features_file, delimiter = ',')
class_labels_pd = pd.read_csv(labels_file, header=None)
class_labels = class_labels_pd[0].tolist()
#params
run_name= name_of_run

# if snakemake.params['scDINO_full_pipeline']:
#     save_dir = snakemake.params['save_dir'][0]
#     neighbors = snakemake.config['downstream_analyses']['kNN']['global']['n_neighbors']
#     temperature = snakemake.config['downstream_analyses']['kNN']['global']['temperature']
# else:

save_dir = save_dir
neighbors = config['downstream_analyses']['kNN']['global']['n_neighbors']
print('NEIGHBORS = ')
print(neighbors)

temperature = config['downstream_analyses']['kNN']['global']['temperature']
print(temperature)

def number_of_classes(labels):
    return len(set(labels))

def label_to_number(labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels_numbers = le.transform(labels)
    return labels_numbers

indices = np.arange(features.shape[0])
np.random.seed(config['meta']['seed'])
np.random.shuffle(indices)
features = features[indices]
class_labels = np.array(class_labels)[indices]
train_features = torch.from_numpy(features[:int(features.shape[0]*0.8),:])
train_labels = torch.from_numpy(label_to_number(class_labels[:int(features.shape[0]*0.8)]))
test_features = torch.from_numpy(features[int(features.shape[0]*0.8):,:])
test_labels = torch.from_numpy(label_to_number(class_labels[int(features.shape[0]*0.8):]))
num_classes = number_of_classes(class_labels)

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes):
    top1, top2, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top2 = top2 + correct.narrow(1, 0, min(2, k)).sum().item()  #dependent on number of classes
        total += targets.size(0)
        top1 = top1 * 100.0 / total
        top2 = top2 * 100.0 / total
        return top1, top2

    
knn_output_dir = save_dir+"/kNN/"

if not os.path.exists(knn_output_dir):
    os.makedirs(knn_output_dir)
with open(knn_output_dir+"global_kNN.txt", 'a') as f:
    f.write("Run: "+run_name+"\n")
    for k in neighbors:
        try:
            top1, top2 = knn_classifier(train_features, train_labels,test_features, test_labels, k, temperature, num_classes=num_classes)
            f.write(f"{k}-NN Top1: {round(top1)}, Top2: {round(top2)}\n")
            #delete variables
            del top1, top2
        except Exception as e:
            f.write("Error: "+str(e)+"\n")             
    f.write("--------------------------------------------------\n")

