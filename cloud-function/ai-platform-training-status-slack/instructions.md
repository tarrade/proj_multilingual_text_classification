# How to deploy cloud functions:

## Publish status of gcloud build on Slack
go to the folder `cloud-function/ai-platform-training-status-slack`:  
`gcloud functions deploy statusAIPlatformTrainingJob \` 
`--entry-point=processLogEntry \`   
`--region=europe-west1 \`      
`--trigger-topic ai-platform-training-job  \`   
`--runtime nodejs10 \`     
`--set-env-vars "SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxxx"`  
