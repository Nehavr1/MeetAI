import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Define events
events = [
    {
        "group_name": "Tech Innovators",
        "event_type": "Hackathon",
        "location": "San Francisco, CA",
        "date": "2025-06-15",
        "participants": 150,
        "budget": 50000
    },
    {
        "group_name": "Green Earth Club",
        "event_type": "Tree Planting Drive",
        "location": "Portland, OR",
        "date": "2025-07-10",
        "participants": 50,
        "budget": 3000
    },
    {
        "group_name": "Art Enthusiasts",
        "event_type": "Art Exhibition",
        "location": "New York, NY",
        "date": "2025-08-05",
        "participants": 200,
        "budget": 20000
    },
    {
        "group_name": "Fitness Freaks",
        "event_type": "Marathon",
        "location": "Austin, TX",
        "date": "2025-09-12",
        "participants": 500,
        "budget": 10000
    },
    {
        "group_name": "Music Lovers",
        "event_type": "Concert",
        "location": "Los Angeles, CA",
        "date": "2025-10-25",
        "participants": 1000,
        "budget": 75000
    }
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SentenceTransformer model initialization
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=str(device))

# Prepare event embeddings
event_texts = [
    f"{e['group_name']} organizes a {e['event_type']} in {e['location']} on {e['date']}."
    for e in events
]
event_embeddings = embedding_model.encode(event_texts, convert_to_tensor=True)

# Hugging Face text generation pipeline
generator = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)

# Class for event retrieval and response generation
class EventAgent:
    def __init__(self, events, event_embeddings, embedding_model, generator):
        self.events = events
        self.event_embeddings = event_embeddings
        self.embedding_model = embedding_model
        self.generator = generator

    def encode_query(self, query):
        return self.embedding_model.encode(query, convert_to_tensor=True)

    def retrieve_events(self, query_embedding, top_k=3):
        cos_scores = util.cos_sim(query_embedding, self.event_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        return top_results

    def prepare_context(self, top_results, max_context_length=100):
        retrieved_contexts = []
        for score, idx in zip(top_results.values, top_results.indices):
            event = self.events[idx]
            retrieved_context = (
                f"Event: {event['event_type']} | Location: {event['location']} | "
                f"Date: {event['date']} | Participants: {event['participants']} | Budget: ${event['budget']}."
            )
            if len(retrieved_context) > max_context_length:
                retrieved_context = retrieved_context[:max_context_length] + "..."
            retrieved_contexts.append(retrieved_context)
        return retrieved_contexts

    def generate_response(self, query, context):
        prompt = (
            f"You are an assistant providing helpful event summaries.\n\n"
            f"User query: {query}\n\n"
            f"Relevant events retrieved:\n{context}\n\n"
            "Generate a concise summary as a bulleted list, with each bullet describing one event clearly and directly answering the query:\n"
            "- "
        )
        response = self.generator(
            prompt,
            max_length=200,
            num_return_sequences=1,
            repetition_penalty=1.2,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.generator.tokenizer.eos_token_id
        )
        return response[0]['generated_text']

    def run(self, query, top_k=3):
        query_embedding = self.encode_query(query)
        top_results = self.retrieve_events(query_embedding, top_k=top_k)
        contexts = self.prepare_context(top_results)
        context_str = "\n".join(contexts)
        response = self.generate_response(query, context_str)
        return contexts, response


# Utility functions for query count display
def stars_for_count(count):
    return "⭐" * min(count, 5)

def get_query_level(count):
    if count < 3:
        return "Base Level"
    elif count < 6:
        return "Mid Level"
    else:
        return "Startup Level"


# Main application loop
def main():
    print("🔍 Welcome to the RAG-Enhanced Event Finder!")
    agent = EventAgent(events, event_embeddings, embedding_model, generator)

    query_counts = {}

    while True:
        user_query = input("\nEnter your search query (or type 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            print("👋 Goodbye!")
            break

        # Update query count
        query_counts[user_query] = query_counts.get(user_query, 0) + 1
        count = query_counts[user_query]
        stars = stars_for_count(count)
        level = get_query_level(count)

        try:
            top_k = int(input("Enter the number of top results to retrieve (default 3): ") or 3)
        except ValueError:
            print("Invalid input. Defaulting to 3 results.")
            top_k = 3

        retrieved_contexts, generated_response = agent.run(user_query, top_k=top_k)

        print(f"\n⭐ Query asked {count} times ({stars}) → {level}")
        print("\n🎯 Relevant Events Retrieved:")
        for idx, ctx in enumerate(retrieved_contexts, 1):
            print(f"{idx}. {ctx}")

        print("\n💡 Generated Response:")
        print(generated_response.strip())


if __name__ == "__main__":
    main()