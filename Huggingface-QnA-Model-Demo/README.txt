Demos how to use a model from hugging face 
Business:
FreshBite Organics is a small D2C business selling organic packaged foods online and via local stores. The team has 12 employees and limited tech expertise.
Current Challenge:
Customers email and message product-related questions (e.g., ingredients, shipping time, shelf life, allergens). The small support team is overwhelmed.

Goal: Automate First-Level Customer Support
We want to deploy an AI assistant trained to answer product FAQs based on our own documentation and FAQs.

USes Roberta-base-squad2 model
Uses freshbite_faq.txt as context
Sets up a small web page using tornado and index.html to respond to queries

Run the python3.9 model-app.py
On browser goto http://127.0.0.1:8686
Ensure all the files (py, txt and html) are in the same folder (keep it in the download folder to keep it simple)
