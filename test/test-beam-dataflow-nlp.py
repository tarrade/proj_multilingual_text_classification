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
from apache_beam.options.pipeline_options import SetupOptions
import src.preprocessing.preprocessing as pp

print(os.environ['PROJECT_ID'])
print(os.environ['BUCKET_NAME'])
print(os.environ['REGION'])

# define query table
def create_query():
    query = """
    SELECT
      id,
      title,
      body,
      tags
    FROM
      `bigquery-public-data.stackoverflow.stackoverflow_posts`
    LIMIT 100
    """

    return query
    
table_schema = {'fields': [
    {'name': 'id', 'type': 'NUMERIC', 'mode': 'REQUIRED'},
    {'name': 'title', 'type': 'STRING', 'mode': 'NULLABLE'},
    {'name': 'text_body', 'type': 'STRING', 'mode': 'NULLABLE'},
    {'name': 'code_body', 'type': 'STRING', 'mode': 'NULLABLE'},
    
    {"fields": [
        {"mode": "NULLABLE", 
         "name": "value", 
         "type": "STRING"}
    ], 
            "mode": "REPEATED", 
            "name": "tags", 
            "type": "RECORD"
    }
]}


def preprocess():
    """
    Arguments:
        -RUNNER: "DirectRunner" or "DataflowRunner". Specfy to run the pipeline locally or on Google Cloud respectively.
    Side-effects:
        -Creates and executes dataflow pipeline.
        See https://beam.apache.org/documentation/programming-guide/#creating-a-pipeline
    """
    job_name = 'test-stackoverflow' + '-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    project = os.environ['PROJECT_ID']
    region = os.environ['REGION']
    output_dir = "gs://{0}/stackoverflow/".format(os.environ['BUCKET_NAME'])

    # options
    options = PipelineOptions()
    google_cloud_options = options.view_as(GoogleCloudOptions)
    google_cloud_options.project =  project
    google_cloud_options.job_name =  job_name
    google_cloud_options.region = region
    google_cloud_options.staging_location = os.path.join(output_dir, 'tmp', 'staging')
    google_cloud_options.temp_location = os.path.join(output_dir, 'tmp')
    # done by command line
    #options.view_as(StandardOptions).runner = RUNNER
    options.view_as(SetupOptions).setup_file=os.environ['DIR_PROJ']+'/setup.py'

    # instantantiate Pipeline object using PipelineOptions
    print('Launching Dataflow job {} ... hang on'.format(job_name))

    p = beam.Pipeline(options=options)
    table = p | 'Read from BigQuery' >> beam.io.Read(beam.io.BigQuerySource(
        # query
        query=create_query(),
        # use standard SQL for the above query
        use_standard_sql=True)
        )
    clean_text = table | 'Clean Text' >> beam.ParDo(pp.NLPProcessing())
    clean_text | 'Write to BigQuery' >> beam.io.WriteToBigQuery(
        # The table name is a required argument for the BigQuery
        table='test_stackoverflow_beam_nlp',
        dataset='test',
        project=project,
        # Here we use the JSON schema read in from a JSON file.
        # Specifying the schema allows the API to create the table correctly if it does not yet exist.
        schema=table_schema,
        # Creates the table in BigQuery if it does not yet exist.
        create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
        # Deletes all data in the BigQuery table before writing.
        write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE)
        # not needed, from with clause

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
# python3 test-beam-dataflow.py --runner DataflowRunner
# python3 test-beam-dataflow.py
# python3 test-beam-dataflow.py --runner DataflowRunner --no_use_public_ips --subnetwork https://www.googleapis.com/compute/v1/projects/xxx/regions/europe-west1/subnetworks/yyyy --region=europe-west1 --zone=europe-west1-b
