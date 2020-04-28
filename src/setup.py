from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['transformers==2.8.0',
                     'google-cloud-bigquery==1.24.0',
                     'google-cloud-storage==1.26.0',
                     'google-cloud-translate==2.0.1',
                     'google-resumable-media== 0.5.0',
                     'cloudml-hypertune==0.1.0.dev6']
setup(
    name='bert_model',
    version='0.1',
    author = 'F. Tarrade',
    author_email = 'fabien.tarrade@gmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Classification of text using BERT'
)