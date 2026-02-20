"""
DSA 2026 Entry Assessment - Section 2: NLP & LLMs
Solutions for Questions N1 to N3
"""

import string

# =============================================================================
# QUESTION N1 — Word Frequency Counter
# =============================================================================

def count_words(text):
    """
    Count word frequencies in a text string.

    Parameters:
    text (str): Input text string

    Returns:
    dict: Dictionary with words as keys and frequencies as values
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts

# Test the function
test_text = "Data Science Africa 2026 is happening in Kampala at Makerere University. Makerere University is a great place for learning."
result = count_words(test_text)
print(result)


# =============================================================================
# QUESTION N2 — Extract Entity Sentences
# =============================================================================

def extract_entities(text, keywords):
    """
    Extract sentences containing specified keywords.

    Parameters:
    text (str): Input text string
    keywords (list): List of keywords to search for

    Returns:
    list: List of sentences containing the keywords
    """
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    matching_sentences = [
        s for s in sentences
        if any(keyword in s for keyword in keywords)
    ]
    return matching_sentences

# Test the function
test_text = "DSA 2026 will be held at Makerere University in Kampala. Kampala is a beautiful city. Makerere University is one of the oldest universities in Africa."
keywords = ["Makerere", "Kampala", "DSA"]
result = extract_entities(test_text, keywords)
print(result)


# =============================================================================
# QUESTION N3 — Jaccard Similarity
# =============================================================================

def calculate_similarity(text1, text2):
    """
    Calculate Jaccard similarity between two text strings.

    Parameters:
    text1 (str): First text string
    text2 (str): Second text string

    Returns:
    float: Jaccard similarity score between 0 and 1
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1 & words2
    union = words1 | words2
    if len(union) == 0:
        return 0.0
    similarity = len(intersection) / len(union)
    return similarity

# Test the function
text1 = "Data Science Africa 2026 Kampala Makerere University"
text2 = "Data Science Africa Kampala Makerere"
similarity = calculate_similarity(text1, text2)
print(f"Similarity: {similarity:.4f}")
