RAG demo that chunks data from freshbite_faq.txt and stores in faiss vector db
Uses Roberta-Base-Squad2 as the AI model
Output is very poor - chunking process and retrieval needs optimization 
Also Roberta is not a very powerful model

Sets up a small web page using tornado and index.html to respond to queries

Run the python3.9 model-app.py
On browser goto http://127.0.0.1:8686
Ensure all the files (py, txt and html) are in the same folder (keep it in the download folder to keep it simple)
