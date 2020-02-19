from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['scikit-learn==0.20.2',
                     'numpy==1.17.2',
                     'google-cloud-bigquery==1.22.0',
                     'google-cloud-dns==0.31.0',
                     'google-cloud-resource-manager==0.30.0',
                     'google-cloud-speech==1.3.1',
                     'google-cloud-vision==0.41.0',
                     'google-cloud-firestore==1.6.0',
                     'google-cloud-bigtable==1.2.0',
                     'google-cloud-logging==1.14.0',
                     'google-cloud-storage==1.23.0',
                     'google-cloud-pubsub==1.0.2',
                     'google-cloud-translate==2.0.0',
                     'google-cloud-spanner==1.13.0',
                     'google-cloud-error-reporting==0.33.0',
                     'google-cloud-datastore==1.10.0',
                     'google-cloud-monitoring==0.34.0',
                     'google-cloud==0.34.0',
                     'google-cloud-trace==0.23.0',
                     'google-api-core==1.14.3',
                     'cloudml-hypertune==0.1.0.dev6',
                     'psutil==5.6.7']

setup(
    name='model',
    version='0.1',
    author = 'F. Tarrade',
    author_email = 'fabien.tarrade@gmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Classification of Stackoverflow post using scikit-learn on GCP'
)