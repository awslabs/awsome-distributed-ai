#!/usr/bin/env python3

import argparse
import jinja2
import os
import pathlib


def get_model_parameters(model_name):
    f = open('models/' + model_name + '.txt')
    return f.read()


def list_models(path='models'):
    models = [str(pathlib.Path(i).with_suffix('')) for i in os.listdir(path)]

    return models


def create_sbatch_file(model_name, model_parameters):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader('slurm'))
    template = env.get_template('training-sub.template')
    content = template.render(MODEL_NAME=model_name,
                              MODEL_PARAMETERS=model_parameters)

    f = open('slurm/' + model_name + '-training.sbatch', mode='w')
    f.write(content)
    f.close()


def create_kubernetes_file(model_name, model_parameters):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader('kubernetes'))
    template = env.get_template('training_kubernetes.template')
    model_name_dashed = model_name.replace('_', '-')
    content = template.render(MODEL_NAME=model_name_dashed,
                              MODEL_PARAMETERS=model_parameters.splitlines())

    f = open('kubernetes/' + model_name + '-fsdp.yaml', mode='w')
    f.write(content)
    f.close()


def main():
    models = list_models()

    for i in models:
        model_parameters = get_model_parameters(i)
        create_sbatch_file(i, model_parameters)
        create_kubernetes_file(i, model_parameters)


if __name__ == '__main__':
    main()