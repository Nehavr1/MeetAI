# MeetAI
This app is used by people who have already created user profiles for groups on meeting apps( like groups for hiking, travelling, excercise etc) 
and new members can join the group using **API queries** to join the relevant group. Once the new members join the group 
the groups increase in the number of followers (or in this case stars). It works on 3 levels. **Base level** is below 
a certain number of followers (in this case below 3 stars), **Midlevel** is above a certain number of followers but below 
a certain higher number of follower( in this case above 3 stars and below 6 stars) and **StartUp** level is over a certain 
number of followers (Over 6 starts). This helps people put effort to build groups and interact better in real time.

Installation files: 
Transformers
Sentense-transformers
Install and import torch
Answer to queries

The app works on
1. Collecting documents: Collects Data on event type, location, date, time
2. Custom Search tool: Search to find the queries like ' where is music event located?'
3. Sematic Search to find the nearest number for similarity: Internally provides a number close to the data
4. Data Chunking: Chunks data based on Location/event/Date
5. Data embedding: Embeddes as a file 
6. Data Retrieval: Embedded file is retrieved for faster processing
7. Indexing: For faster data retreval
8. LLM EventAgent from Huggingface: LLM is used to provide answers like human
9. RAG generation: Information is provided from internet
10. Function Calling: For easier access to functions when the data is large
11. Prompt enginnering
12. MultiStep Agent: Agent helps in mutistaging process like finding events, finding location
13. Github: Code uploaded
14. Streamlit: Code publised 

15. ![image](https://github.com/user-attachments/assets/6ae311aa-2211-4ea1-996f-49e08145f74d)





