# Grammar Feedback API

Back end application that performs the following tasks:

- Splits the sentences in an input text.
- Performs grammatical error correction using an LLM.
- Uses ERRANT to extract edits from the sentence pair.
- Generates feedback for each edit.

Currently, there are two feedback systems supported, `rule-based` and `llm-based`:

- `rule-based` operates directly on the ERRANT edits, filling simple templates.
- `llm-based` first uses an LLM to "refine" the edits to address corner cases and break down multiple changes on a given word. Then, it generates feedback using another LLM prompt.

## Launching

### Local

```bash
gunicorn -w 1 -k uvicorn.workers.UvicornWorker application:application --bind 0.0.0.0:8000 
```

Recommended Environmental Variables:

```bash
ENABLE_FILE_LOGGING="true"
ALLOWED_ORIGINS="http://localhost,http://localhost:3000"
```

### AWS (Elastic Beanstalk)

- Create an AWS account and complete any initial steps.

- For those without existing experience, completing [the introduction to Elastic Beanstalk here](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/GettingStarted.html) is recommended as well.

- Create an application for this program on Elastic Beanstalk.

- [Install the EB CLI](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-install.html) and create an environment for this program:

```bash
eb init
eb create grammar-feedback-api-env
eb deploy
```

- Enable the following trust relationship in the environment's service role (by default, `aws-elasticbeanstalk-service-role`):

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "",
            "Effect": "Allow",
            "Principal": {
                "Service": "elasticbeanstalk.amazonaws.com"
            },
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {
                    "sts:ExternalId": "elasticbeanstalk"
                }
            }
        }
    ]
}
```

- In the "Configure instance traffic and scaling" section of the environment's configuration, set the health check path to `/health`.

- Using the S3 analysis saving functions will require you to set up a bucket and add the following inline permission to the environment's instance profile role (by default, `aws-elasticbeanstalk-ec2-role`):

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject"
            ],
            "Resource": "arn:aws:s3:::your-s3-bucket/*"
        }
    ]
}
```

- In the "Configure updates, monitoring, and logging" section of the environment's configuration, set environmental variables:

```bash
OPENAI_API_KEY="your-api-key"
AWS_REGION="your-region"
SAVE_TO_S3="true"
S3_BUCKET_NAME="your-s3-bucket"
ALLOWED_ORIGINS="your-front-end"
```

## Usage

```python
import requests
import json

url = "http://0.0.0.0:8000/grammar_feedback"

text = "This am a sentence. This is second santence."

payload = {
    "text": text,
    "user_id": "some_uuid",
    "system_choice": "rule-based", #or "llm-based"
    "draft_number": 1,
}

response = requests.post(url, json=payload)
print("Response Code:", response.status_code)
print()
if response.status_code == 200:
    data = response.json()
    feedback_list = data.get("feedback_list", [])

    for feedback in feedback_list:
        highlighted_source = feedback["source"][:feedback["highlight_start"]] + "<" + feedback["highlight_text"] + ">" + feedback["source"][feedback["highlight_end"]:]
        print(f"Sentence:       {highlighted_source}")
        print(f"Error Tag:      {feedback['error_tag']}")
        print(f"What's wrong?   {feedback['feedback_explanation']}")
        print(f"What to do:     {feedback['feedback_suggestion']}")
        print()
```

Output:

```output
Response Code: 200

Sentence:       This <am> a sentence.
Error Tag:      Subject-Verb Agreement
What's wrong?   The form of the verb "am" does not agree with the subject of the sentence.
What to do:     Change "am" to a form that agrees with the subject.

Sentence:       This <is second> santence.
Error Tag:      Determiner
What's wrong?   Something seems to be missing in this part of the sentence.
What to do:     Consider inserting "the" here.

Sentence:       This is second <santence>.
Error Tag:      Spelling
What's wrong?   "santence" may be mispelled.
What to do:     Consider changing "santence" to "sentence."
```
