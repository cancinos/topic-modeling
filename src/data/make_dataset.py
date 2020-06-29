# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import re
import pickle
from pprint import pprint
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.utils import text_processing


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df = pd.read_json(input_filepath)
    data = df.content.values.tolist()
    data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data] # Eliminate emails in data
    data = [re.sub(r'\s+', ' ', sent) for sent in data] # Eliminate new lines in data
    data = [re.sub(r"\'", "", sent) for sent in data] # Eliminate ''
    words = list(text_processing.sentences_to_words(data))
    with open(output_filepath, 'wb') as file:
        pickle.dump(words, file)
        #pd.DataFrame(words).to_json(file, force_ascii=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
