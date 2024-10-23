# POS_tagging

def train_bigram_hmm(sentences):
    """ Trains the HMM using the Bigram model and returns the parameters π (start probabilities), A (transition probabilities), and B (emission probabilities). """
    tags = {tag for sentence in sentences for _, tag in sentence}
    words = {word for sentence in sentences for word, _ in sentence}

    # Calculate start probabilities π
    start_tag_count = Counter(sentence[0][1] for sentence in sentences if sentence)
    total_sentences = len(sentences)
    start_probabilities = {tag: start_tag_count[tag] / total_sentences for tag in tags}

    # Calculate transition probabilities A based on tags
    transition_counts = defaultdict(Counter)
    for sentence in sentences:
        previous_tag = None
        for word, current_tag in sentence:
            if previous_tag is not None:
                transition_counts[previous_tag][current_tag] += 1
            previous_tag = current_tag

    transition_probabilities = {
        previous_tag: {current_tag: count / sum(tag_counts.values())
                       for current_tag, count in tag_counts.items()}
        for previous_tag, tag_counts in transition_counts.items()
    }

    # Calculate emission probabilities B based on the current word and tag
    emission_counts = defaultdict(Counter)
    for sentence in sentences:
        for word, tag in sentence:
            emission_counts[tag][word] += 1

    emission_probabilities = {
        tag: {word: count / sum(word_counts.values())
              for word, count in word_counts.items()}
        for tag, word_counts in emission_counts.items()
    }

    return start_probabilities, transition_probabilities, emission_probabilities

