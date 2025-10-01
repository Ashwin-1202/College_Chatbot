import json
import random
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import os
import nltk
try:
    nltk.data.find('tokenizers/punkt')  # Check if punkt tokenizer exists
except LookupError:
    nltk.download('punkt')  # Download if missing

try:
    nltk.data.find('corpora/stopwords')  # Check if stopwords exist
except LookupError:
    nltk.download('stopwords')  # Download if missing
# Load knowledge base
try:
    with open('college_data.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    print("✓ College knowledge base loaded successfully")
    print(f"✓ Loaded {len(data['intents'])} intents")
    
    # Debug: Show available intents
    print("📋 Available intents:")
    for intent in data['intents']:
        patterns_sample = intent['patterns'][:3]  # Show first 3 patterns
        print(f"   - {intent['tag']}: {len(intent['patterns'])} patterns (e.g., {patterns_sample})")
        
except Exception as e:
    print(f"✗ Error loading JSON: {e}")
    data = {"intents": []}

# Enhanced keyword mapping with priority scoring
keyword_mapping = {
    # Fees and payments (priority: 3)
    'fee': ('fee_structure', 3), 'fees': ('fee_structure', 3), 'payment': ('fee_structure', 3), 
    'tuition': ('fee_structure', 3), 'cost': ('fee_structure', 3), 'money': ('fee_structure', 2),
    
    # Transportation (priority: 2)
    'bus': ('transportation', 3), 'transport': ('transportation', 3), 'travel': ('transportation', 2),
    'commute': ('transportation', 2), 'route': ('transportation', 2), 'vehicle': ('transportation', 1),
    
    # Placements (priority: 3)
    'placement': ('placement_info', 3), 'job': ('placement_info', 3), 'company': ('placement_info', 2),
    'recruitment': ('placement_info', 2), 'career': ('placement_info', 2), 'salary': ('placement_info', 2),
    'package': ('placement_info', 2), 'hire': ('placement_info', 1),
    
    # Courses (priority: 2)
    'course': ('course_information', 3), 'program': ('course_information', 2), 
    'subject': ('course_information', 2), 'syllabus': ('course_information', 2),
    'curriculum': ('course_information', 2), 'study': ('course_information', 1),
    
    # Faculty (priority: 2)
    'faculty': ('faculty_contacts', 3), 'professor': ('faculty_contacts', 2), 
    'teacher': ('faculty_contacts', 2), 'staff': ('faculty_contacts', 1),
    'hod': ('faculty_contacts', 3), 'department': ('faculty_contacts', 1),
    
    # Library (priority: 2)
    'library': ('library_details', 3), 'book': ('library_details', 2), 
    'borrow': ('library_details', 2), 'research': ('library_details', 1),
    
    # Facilities (priority: 3) - Enhanced with more keywords
    'facility': ('facilities', 3), 'facilities': ('facilities', 3), 'hostel': ('facilities', 3),
    'sports': ('facilities', 2), 'lab': ('facilities', 2), 'laboratory': ('facilities', 2),
    'canteen': ('facilities', 2), 'cafeteria': ('facilities', 2), 'gym': ('facilities', 2),
    'medical': ('facilities', 2), 'campus': ('facilities', 1), 'infrastructure': ('facilities', 2),
    'accommodation': ('facilities', 2), 'ground': ('facilities', 1), 'auditorium': ('facilities', 1),
    
    # Admissions (priority: 3)
    'admission': ('admission_process', 3), 'apply': ('admission_process', 2),
    'application': ('admission_process', 2), 'eligibility': ('admission_process', 2),
    'entrance': ('admission_process', 2), 'admit': ('admission_process', 1),
    
    # Scholarships (priority: 2)
    'scholarship': ('scholarships', 3), 'financial': ('scholarships', 2),
    'aid': ('scholarships', 2), 'loan': ('scholarships', 1),
    
    # Events (priority: 1)
    'event': ('events_clubs', 2), 'fest': ('events_clubs', 2), 'club': ('events_clubs', 2),
    'activity': ('events_clubs', 1), 'workshop': ('events_clubs', 1)
}

def preprocess_text(text):
    """Enhanced preprocessing that keeps the meaning intact"""
    if not text:
        return ""
    
    text = text.lower().strip()
    
    # Remove punctuation but keep important symbols
    text = re.sub(r'[^\w\s?]', '', text)
    
    # Remove common question words that don't add meaning (shorter list)
    question_words = ['what', 'where', 'when', 'why', 'how', 'can', 'could', 'would', 'should', 'tell', 'give', 'me']
    words = [word for word in text.split() if word not in question_words and len(word) > 1]
    
    return ' '.join(words)

def get_best_keyword_match(user_input):
    """Enhanced keyword matching with priority scoring"""
    user_lower = user_input.lower()
    user_words = user_lower.split()
    
    keyword_scores = {}
    
    for word in user_words:
        if word in keyword_mapping:
            tag, priority = keyword_mapping[word]
            if tag not in keyword_scores:
                keyword_scores[tag] = 0
            keyword_scores[tag] += priority
    
    if keyword_scores:
        # Return the tag with highest score
        best_tag = max(keyword_scores.items(), key=lambda x: x[1])[0]
        return best_tag
    
    return None

def get_response(user_input):
    """Enhanced main function to get bot response"""
    if not user_input or not user_input.strip():
        return "Please ask me something about the college!"
    
    user_lower = user_input.lower().strip()
    user_clean = preprocess_text(user_input)
    
    # Quick responses for very short queries
    if len(user_lower) <= 3:
        if user_lower in ['hi', 'hey']:
            return "Hello! Welcome to College Helpdesk! How can I assist you today?"
        elif user_lower in ['bye', 'exit']:
            return "Goodbye! Have a great day! 👋"
    
    # Method 1: Enhanced Keyword matching with priority (fastest)
    best_tag = get_best_keyword_match(user_input)
    if best_tag:
        for intent in data['intents']:
            if intent['tag'] == best_tag:
                return random.choice(intent['responses'])
    
    # Method 2: Direct pattern matching (exact and partial matches)
    for intent in data['intents']:
        for pattern in intent['patterns']:
            pattern_lower = pattern.lower()
            # Check for exact match or significant overlap
            if (user_lower in pattern_lower or 
                pattern_lower in user_lower or
                any(word in user_lower for word in pattern_lower.split() if len(word) > 3)):
                return random.choice(intent['responses'])
    
    # Method 3: TF-IDF similarity (for complex queries)
    if data['intents'] and user_clean:
        all_patterns = []
        all_tags = []
        
        for intent in data['intents']:
            for pattern in intent['patterns']:
                processed_pattern = preprocess_text(pattern)
                if processed_pattern:  # Only add non-empty patterns
                    all_patterns.append(processed_pattern)
                    all_tags.append(intent['tag'])
        
        if all_patterns:
            try:
                vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(all_patterns)
                input_vector = vectorizer.transform([user_clean])
                similarities = cosine_similarity(input_vector, tfidf_matrix)
                
                if similarities.size > 0:
                    best_match_idx = np.argmax(similarities)
                    best_score = similarities[0, best_match_idx]
                    
                    if best_score > 0.15:  # Lower threshold for better matching
                        best_tag = all_tags[best_match_idx]
                        for intent in data['intents']:
                            if intent['tag'] == best_tag:
                                return random.choice(intent['responses'])
            except Exception as e:
                # Silently fail and continue to fallback
                pass
    
    # Smart fallback based on detected keywords
    detected_keywords = []
    for word in user_lower.split():
        if word in keyword_mapping:
            detected_keywords.append(keyword_mapping[word][0])
    
    if detected_keywords:
        # Suggest based on detected but unmatched keywords
        unique_keywords = list(set(detected_keywords))
        suggestions = " or ".join(unique_keywords[:2])  # Show top 2 suggestions
        return f"I can help you with {suggestions}. Could you rephrase your question?"
    
    # Context-aware fallback responses
    if any(word in user_lower for word in ['hi', 'hello', 'hey', 'good morning', 'good afternoon']):
        return random.choice([
            "Hello! Welcome to College Helpdesk! How can I assist you today?",
            "Hi there! I'm here to help with college information. What do you need to know?",
            "Greetings! Ask me about admissions, courses, fees, placements, or campus facilities."
        ])
    
    if any(word in user_lower for word in ['bye', 'goodbye', 'see you', 'exit']):
        return "Thank you for chatting! Feel free to ask if you have more questions. Have a great day! 👋"
    
    if any(word in user_lower for word in ['thank', 'thanks']):
        return "You're welcome! Is there anything else I can help you with?"
    
    # Final fallback with suggestions
    fallback_suggestions = [
        "I can help with: Admissions, Courses, Fees, Placements, Facilities, Faculty, Library, or Transportation. What would you like to know?",
        "Try asking about: • Fee structure • Placement statistics • Campus facilities • Admission process • Faculty contacts",
        "I specialize in college information. You can ask me about fees, placements, hostels, labs, sports facilities, or any other campus-related queries."
    ]
    
    return random.choice(fallback_suggestions)

# Conversation memory for better context
conversation_context = {}

def get_response_with_context(user_input, user_id="default"):
    """Enhanced version with conversation memory"""
    global conversation_context
    
    if user_id not in conversation_context:
        conversation_context[user_id] = {
            'last_topic': None,
            'question_count': 0,
            'asked_about': []
        }
    
    context = conversation_context[user_id]
    context['question_count'] += 1
    
    # Check for follow-up questions
    if context['last_topic'] and any(word in user_input.lower() for word in ['more', 'detail', 'another', 'else', 'what about', 'and']):
        for intent in data['intents']:
            if intent['tag'] == context['last_topic']:
                follow_ups = [
                    "Here's more information:",
                    "Additional details:",
                    "Also, you might want to know:",
                    "More about that:"
                ]
                return f"{random.choice(follow_ups)}\n{random.choice(intent['responses'])}"
    
    # Get regular response
    response = get_response(user_input)
    
    # Update context with current topic
    for intent in data['intents']:
        if any(pattern.lower() in user_input.lower() for pattern in intent['patterns']):
            context['last_topic'] = intent['tag']
            if intent['tag'] not in context['asked_about']:
                context['asked_about'].append(intent['tag'])
            break
        # Also update based on keyword matches
        best_tag = get_best_keyword_match(user_input)
        if best_tag and best_tag == intent['tag']:
            context['last_topic'] = intent['tag']
            if intent['tag'] not in context['asked_about']:
                context['asked_about'].append(intent['tag'])
            break
    
    return response

# Only run terminal version if this file is executed directly
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🎓 Enhanced College Helpdesk AI Assistant")
    print("Type 'quit' to exit")
    print("="*50)
    
    # Test the facilities intent specifically
    test_queries = [
        "facilities",
        "hostel",
        "sports facilities", 
        "campus facilities",
        "what facilities are available"
    ]
    
    print("\n🧪 Testing facilities queries:")
    for query in test_queries:
        response = get_response(query)
        print(f"Q: {query}")
        print(f"A: {response}\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("College Bot: Goodbye! 👋")
                break
                
            response = get_response_with_context(user_input)
            print(f"College Bot: {response}")
            
        except KeyboardInterrupt:
            print("\nCollege Bot: Session ended. Have a great day! 👋")
            break
        except Exception as e:
            print(f"College Bot: Sorry, I encountered an error. Please try again.")