from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['scikit-learn==0.20',
                     'numpy>=1.14.0',
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
                     'cloudml-hypertune',
                     'psutil']

#setup(name='trainer',
#      version='1.0',
#      description='Stackoverflow with scikit-learn',
#      author='Google',
#      author_email='nobody@google.com',
#      license='Apache2',
#      packages=['trainer'],
#      ## WARNING! Do not upload this package to PyPI
#      ## BECAUSE it contains a private key
#      package_data={'': ['privatekey.json']},
#      install_requires=[
#          'pandas-gbq==0.3.0',
#          'urllib3',
#          'google-cloud-bigquery==0.29.0',
#          'cloudml-hypertune'
#      ],
#      zip_safe=False)

setup(
    name='trainer',
    version='0.1',
    author = 'F. Tarrade',
    author_email = 'fabien.tarrade@gmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Classification of Stackoverflow post using scikit-learn on GCP'
)