const { IncomingWebhook } = require('@slack/webhook');

// Webhook for Slack
const url = process.env.SLACK_WEBHOOK_URL;
const webhook = new IncomingWebhook(url);

module.exports.processLogEntry= (pubSubEvent, context) => {
  const build = eventToBuild(pubSubEvent.data);
  //console.log(build);
  //console.log(`Id: ${build.resource.labels.job_id}`);
  //console.log(`Payload: ${build.textPayload}`);
  //console.log(`Timestamp: ${build.timestamp}`);
  var job_status = 'Failed';
  const job_id = build.resource.labels.job_id;
  const payload = build.textPayload;
  const timestamp = build.timestamp;
  const job_type = build.resource.type;
  if (payload.includes('successfully')){
    job_status = 'Succeed';
  }
  // Send message to Slack.
  const message = createSlackMessage(job_status, job_id, payload, timestamp, job_type);
  webhook.send(message);
};

// eventToBuild transforms pubsub event message to a build object.
const eventToBuild = (data) => {
  return JSON.parse(Buffer.from(data, 'base64').toString());
}

// createSlackMessage creates a message from a build object.
const createSlackMessage = (job_status, job_id, payload, timestamp, job_type) => {
  // template to write message in Slack
  const emoticon = job_status=='Failed' ? ':x:' : ':white_check_mark:';
  notification = `${emoticon} Job: ${job_id}\nStatus:  ${job_status} \nType: ${job_type}\nTimestamp: ${timestamp}\nDetails ${payload}`
  console.log('sending a Slack message');
  const message = {
    text: notification
  };
  return message;
}