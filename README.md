# Introducing the World's Greatest Homebrewed LLM!

It sometimes works!

### Using this model

This model includes the Amazon Topical-Chat dataset, which is intended to be used alongside web-queried documents (e.g., from Wikipedia). This model requires inputs to be of the form:

((input_message, input_document), output_msg), where:
 * "input_msg" is an agent message (generally, the "user message");
 * "input_doc" is an additional message containing relevant (usually web-queried) data;
 * "output_msg" is an agent message (generally, the target response to the input message)

This repository contains a sample chroma database, which contains a myriad of Wikipedia articles by topic that is queried by similarity to an input document. Though not required, it is highly
recommended to use a persistent chroma instance to store sentence embeddings for the similarity search, as this will result in Wikipedia being queried directly (which is fine, if you like to watch 
paint dry).

If you want the web page demo for this LLM, then ~~you can find it here~~ you're just gonna have to wait.

### Why have you done this?

For kicks.

## Legal Stuff

NOTICE: This project uses data available in Amazon's Topical-Chat repository (https://github.com/alexa/Topical-Chat). This project uses modified portions of the conversational training dataset, which are saved to the directory "./model_tensors". This repository will be updated to contain an unmodified copy of the original data license (named DATALICENSE), which can also be found here: https://github.com/alexa/Topical-Chat/blob/master/DATALICENSE
