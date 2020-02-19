import sys
import os
import pathlib
import logging
import subprocess
import datetime
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import WorkerOptions
from apache_beam.options.pipeline_options import SetupOptions
import src.preprocessing.beam_dofn as pp

print(os.environ['PROJECT_ID'])
print(os.environ['BUCKET_NAME'])
print(os.environ['REGION'])

# define query table
def data_query():
    query = """
SELECT
  id,
  title,
  body,
  tags
FROM
  `nlp-text-classification.stackoverflow.posts_p1`
"""
    return query

def tag_query():
    query = """
    SELECT
      tag
    FROM
      `nlp-text-classification.stackoverflow.tags`
    WHERE
      tag <> ''
    ORDER BY
      count DESC
    LIMIT
      50
    """
    return query
    
table_schema = {'fields': [
    {'name': 'id', 'type': 'NUMERIC', 'mode': 'REQUIRED'},
    {'name': 'title', 'type': 'STRING', 'mode': 'NULLABLE'},
    {'name': 'text_body', 'type': 'STRING', 'mode': 'NULLABLE'},
    {'name': 'code_body', 'type': 'STRING', 'mode': 'NULLABLE'},
    {'name': 'tags', 'type': 'STRING', 'mode': 'REPEATED'},
]}


def preprocess():
    """
    Arguments:
        -RUNNER: "DirectRunner" or "DataflowRunner". Specfy to run the pipeline locally or on Google Cloud respectively.
    Side-effects:
        -Creates and executes dataflow pipeline.
        See https://beam.apache.org/documentation/programming-guide/#creating-a-pipeline
    """
    job_name = 'stackoverflow-raphael' + '-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    project = os.environ['PROJECT_ID']
    region = os.environ['REGION']
    output_dir = "gs://{0}/".format(os.environ['BUCKET_NAME'])

    #options
    options = PipelineOptions()
    
    google_cloud_options = options.view_as(GoogleCloudOptions)
    google_cloud_options.project =  project
    google_cloud_options.region = region
    google_cloud_options.job_name =  job_name
    google_cloud_options.staging_location = os.path.join(output_dir, 'beam', 'stage')
    google_cloud_options.temp_location = os.path.join(output_dir, 'beam', 'temp')
    
    worker_options = options.view_as(WorkerOptions)
    worker_options.max_num_workers =  100
    worker_options.zone = 'europe-west6-b'
    worker_options.use_public_ips=False
    worker_options.network = 'default'
   # worker_options.disk_size_gb = 50

    #options.view_as(StandardOptions).runner = RUNNER
    options.view_as(SetupOptions).setup_file=os.environ['DIR_PROJ']+'/setup.py'

    # instantantiate Pipeline object using PipelineOptions
    print('Launching Dataflow job {} ... hang on'.format(job_name))
    
    #table reference
    new_table = beam.io.gcp.internal.clients.bigquery.TableReference(
    projectId='nlp-text-classification',
    datasetId='stackoverflow',
    tableId='posts_preprocessed')
    
    with beam.Pipeline(options=options) as p:
        post_table = p            | "Read Posts from BigQuery" >> beam.io.Read(beam.io.BigQuerySource(
                                                    query=data_query(),
                                                    use_standard_sql=True))
        #tag_table = p             | "Read Tags from BigQuery" >> beam.io.Read(beam.io.BigQuerySource(
                                                    #query=tag_query(),
                                                    #use_standard_sql=True))
        clean_text = post_table   | "Preprocessing" >> beam.ParDo(pp.NLP())
        clean_text                | "Write Posts to BigQuery" >> beam.io.WriteToBigQuery(
                                                    new_table,
                                                    schema=table_schema,
                                                    write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
                                                    create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED)
        str_values = clean_text   | "Post Records to Text" >> beam.ParDo(pp.CSV())

        str_values                | "Write Posts to GCS"  >> beam.io.WriteToText(output_dir+'results/posts',
                                                    file_name_suffix='.csv', 
                                                    header='id, title, text_body, code_body, tags')

    if options.view_as(StandardOptions).runner == 'DataflowRunner':
        print('DataflowRunner')
        p.run()
    else:
        print('Default: DirectRunner')
        result = p.run()
        result.wait_until_finish()
    print('Done')

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    print('Starting main process ...')
    preprocess()

# Usage
# python3 preprocessing.py --runner DataflowRunner
# python3 beam-pipeline.py
# python3 beam-pipeline.py --runner DataflowRunner --no_use_public_ips --subnetwork 