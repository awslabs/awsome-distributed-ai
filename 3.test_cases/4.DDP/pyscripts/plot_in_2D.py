import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
import umap
import ast
import os
import seaborn as sns
import yaml
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


#load data
features = np.genfromtxt(features_file, delimiter = ',')
class_labels_pd = pd.read_csv(labels_file, header=None)
class_labels = class_labels_pd[0].tolist()

#create directories to save plots
#if snakemake.params['scDINO_full_pipeline']:
#    save_dir= f"{snakemake.wildcards['save_dir_downstream_run']}/{snakemake.wildcards['ViT_name']}_channel{snakemake.wildcards['channel_names']}_analyses/embedding_plots/"
#    file_name= f"epoch{snakemake.wildcards['epoch_num']}_"
#    umap_params = snakemake.config['downstream_analyses']['umap_eval']
#else:

dino_vit_name = config['train_scDINO']['dino_vit_name']

selected_channels = [config['meta']['selected_channel_combination_per_run']]
selected_channels = list(ast.literal_eval(selected_channels[0]))

channel_dict = config['meta']['channel_dict']
channel_dict = dict(zip(selected_channels, channel_dict.split(',')))

channel_names=get_channel_name_combi_list(selected_channels, channel_dict)

save_dir = f"{save_dir_downstream_run}/{dino_vit_name}_channel{channel_names[0]}_analyses/embedding_plots/"
file_name = f"{channel_names[0]}_{dino_vit_name}"

#UMAP
def fit_umap(data, n_neighbors, min_dist, metric, spread, epochs):
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, spread=spread, n_epochs=epochs, random_state=42)
    umap_embedding = umap_model.fit_transform(data)
    return umap_embedding

n_neighbors = config['downstream_analyses']['umap_eval']['n_neighbors']
min_dist = config['downstream_analyses']['umap_eval']['min_dist']
metric = config['downstream_analyses']['umap_eval']['metric']
spread = config['downstream_analyses']['umap_eval']['spread']
epochs = config['downstream_analyses']['umap_eval']['epochs']

umap_embedding = fit_umap(features, n_neighbors=15, min_dist=0.1, metric=metric, spread=spread, epochs=epochs)

custom_palette = sns.color_palette("hls", len(set(class_labels)))

def make_plot(embedding, labels, save_dir, file_name=file_name,name="Emb type", description="details"):
    sns_plot = sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=labels, s=14, palette=custom_palette, linewidth=0, alpha=0.9)
    plt.suptitle(f"{name}_{file_name}", fontsize=8)
    sns_plot.tick_params(labelbottom=False)
    sns_plot.tick_params(labelleft=False)
    sns_plot.tick_params(bottom=False)
    sns_plot.tick_params(left=False)
    sns_plot.set_title("CLS Token embedding of "+str(len(labels))+" cells with a dimensionality of "+str(features.shape[1])+" \n"+description, fontsize=6)
    sns.move_legend(sns_plot, "lower left", title='Classes', prop={'size': 5}, title_fontsize=6, markerscale=0.5)
    sns.set(rc={"figure.figsize":(14, 10)})
    sns.despine(bottom = True, left = True)
    sns_plot.figure.savefig(f"{save_dir}{file_name}{name}.png", dpi=325)
    sns_plot.figure.savefig(f"{save_dir}pdf_format/{file_name}{name}.pdf")
    plt.close()

os.makedirs(f"{save_dir}pdf_format", exist_ok=True)

make_plot(umap_embedding, class_labels, save_dir=save_dir, file_name=file_name, name="umap",description=f"n_neighbors:{n_neighbors}, min_dist={min_dist}, metric={metric}, spread={spread}, epochs={epochs}")


########################### Additional plots from https://topometry.readthedocs.io/en/latest/ ###########################
if config['downstream_analyses']['umap_eval']['topometry_plots']:
    
    import topo as tp

    os.makedirs(f"{save_dir}/topometry_plots", exist_ok=True)
    os.makedirs(f"{save_dir}/topometry_plots/pdf_format", exist_ok=True)

    save_dir_topo = f"{save_dir}/topometry_plots/"
    # Learn topological metrics and basis from data. The default is to use diffusion harmonics.
    tg = tp.TopOGraph()

    print('running all combinations')
    tg.run_layouts(features, n_components=2,
                        bases=['diffusion', 'fuzzy'],
                        graphs=['diff', 'fuzzy'],
                        layouts=['tSNE', 'MAP', 'MDE', 'PaCMAP', 'TriMAP', 'NCVis'])

    make_plot(tg.db_diff_MAP, class_labels, name="db_diff_MAP", save_dir=save_dir_topo)
    make_plot(tg.db_fuzzy_MAP, class_labels, name="db_fuzzy_MAP", save_dir=save_dir_topo)
    make_plot(tg.db_diff_MDE, class_labels, name="db_diff_MDE", save_dir=save_dir_topo)
    make_plot(tg.db_fuzzy_MDE, class_labels, name="db_fuzzy_MDE", save_dir=save_dir_topo)
    make_plot(tg.db_PaCMAP, class_labels, name="db_PaCMAP", save_dir=save_dir_topo)
    make_plot(tg.db_TriMAP, class_labels, name="db_TriMAP", save_dir=save_dir_topo)
    make_plot(tg.db_tSNE, class_labels, name="db_tSNE", save_dir=save_dir_topo)
    make_plot(tg.fb_diff_MAP, class_labels, name="fb_diff_MAP", save_dir=save_dir_topo)
    make_plot(tg.fb_fuzzy_MAP, class_labels, name="fb_fuzzy_MAP", save_dir=save_dir_topo)
    make_plot(tg.fb_diff_MDE, class_labels, name="fb_diff_MDE", save_dir=save_dir_topo)
    make_plot(tg.fb_fuzzy_MDE, class_labels, name="fb_fuzzy_MDE", save_dir=save_dir_topo)
    make_plot(tg.fb_PaCMAP, class_labels, name="fb_PaCMAP", save_dir=save_dir_topo)
    make_plot(tg.fb_TriMAP, class_labels, name="fb_TriMAP", save_dir=save_dir_topo)
    make_plot(tg.fb_tSNE, class_labels, name="fb_tSNE", save_dir=save_dir_topo)
