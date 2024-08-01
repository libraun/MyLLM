# Introducing the World's Greatest Homebrewed LLM!

It sometimes works!

### Using this model

This repository includes the Amazon Topical-Chat dataset, which is intended to be used alongside web-queried documents (e.g., relevant passages from Wikipedia). It contains a pretrained encoder and decoder, but if you want to train this model on your own data, ensure that the data is of the form:

((input_message, input_document), output_msg), where:
 * "input_msg" is an agent message (e.g., the "user message");
 * "input_doc" is an additional message containing relevant (web-queried) data;
 * "output_msg" is another agent message (e.g., the target response to the input message)

This repository contains a sample chroma database, which contains a sizeable number of Wikipedia articles that can be queried by similarity to an input document. Though not required, it is highly
recommended to use a persistent Chroma database instance to store sentence embeddings for retrieval, because otherwise Wikipedia will be queried directly.

## Legal Stuff

NOTICE: This project uses data available in Amazon's Topical-Chat repository (https://github.com/alexa/Topical-Chat). This project uses modified portions of the conversational training dataset, which are saved to the directory "./model_tensors". The original, unmodified  data license (named DATALICENSE) for Amazon's Topical-Chat dataset can be found here: https://github.com/alexa/Topical-Chat/blob/master/DATALICENSE
