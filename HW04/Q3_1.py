import os
import math


# Path to the dataset
DATA_PATH = 'languageID'

# Get the list of files for each language
english_files = [f for f in os.listdir(DATA_PATH) if f.startswith('e') and int(f[1:-4]) < 10]
spanish_files = [f for f in os.listdir(DATA_PATH) if f.startswith('s') and int(f[1:-4]) < 10]
japanese_files = [f for f in os.listdir(DATA_PATH) if f.startswith('j') and int(f[1:-4]) < 10]

# Total number of training files
N = len(english_files) + len(spanish_files) + len(japanese_files)

# Calculate the prior probabilities with additive smoothing
alpha = 0.5
num_classes = 3

prior_e = (len(english_files) + alpha) / (N + alpha * num_classes)
prior_s = (len(spanish_files) + alpha) / (N + alpha * num_classes)
prior_j = (len(japanese_files) + alpha) / (N + alpha * num_classes)

# Convert the probabilities to log-space
log_prior_e = math.log(prior_e)
log_prior_s = math.log(prior_s)
log_prior_j = math.log(prior_j)

# Print the results
print(f"Prior probability for English: {prior_e} (log: {log_prior_e})")
print(f"Prior probability for Spanish: {prior_s} (log: {log_prior_s})")
print(f"Prior probability for Japanese: {prior_j} (log: {log_prior_j})")


# Define the characters we're interested in
characters = [chr(i) for i in range(97, 123)] + [' ']

# Initialize a dictionary to count the occurrences of each character in English documents
char_counts = {char: 0 for char in characters}

# Read the English training files and update the char_counts dictionary
for file_name in english_files:
    with open(os.path.join(DATA_PATH, file_name), 'r', encoding='utf-8') as file:
        content = file.read()
        for char in characters:
            char_counts[char] += content.count(char)

# Compute the total number of characters in English training documents
total_chars = sum(char_counts.values())

# Compute the class conditional probabilities for English with additive smoothing
alpha = 0.5
theta_e = {char: (count + alpha) / (total_chars + alpha * 27) for char, count in char_counts.items()}

# Print the results
for char, prob in theta_e.items():
    print(f"Probability of character '{char}' given English: {prob}")


# Function to compute the class conditional probabilities for a given language
def compute_theta(language_files, language_label):
    char_counts = {char: 0 for char in characters}
    
    # Read the language training files and update the char_counts dictionary
    for file_name in language_files:
        with open(os.path.join(DATA_PATH, file_name), 'r', encoding='utf-8') as file:
            content = file.read()
            for char in characters:
                char_counts[char] += content.count(char)
                
    # Compute the total number of characters in the language training documents
    total_chars = sum(char_counts.values())
    
    # Compute the class conditional probabilities for the language with additive smoothing
    theta = {char: (count + alpha) / (total_chars + alpha * 27) for char, count in char_counts.items()}
    
    # Print the results
    print(f"\nClass conditional probabilities for {language_label}:")
    for char, prob in theta.items():
        print(f"Probability of character '{char}' given {language_label}: {prob}")
    
    return theta

# Compute and print class conditional probabilities for Japanese and Spanish
theta_j = compute_theta(japanese_files, 'Japanese')
theta_s = compute_theta(spanish_files, 'Spanish')



def generate_bow_vector(filename):
    # Initialize a dictionary to store counts of each character
    char_counts = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
    
    # Create the full path to the file
    full_path = os.path.join(DATA_PATH, filename)
    
    # Read the file content
    with open(full_path, 'r', encoding='utf-8') as file:
        content = file.read().lower()  # Convert the content to lowercase
    
    # Count each character
    for char in content:
        if char in char_counts:
            char_counts[char] += 1

    # Convert the dictionary to a vector
    bow_vector = [char_counts[char] for char in 'abcdefghijklmnopqrstuvwxyz ']

    return bow_vector

# Generate the BOW vector for e10.txt
e10_bow_vector = generate_bow_vector('e10.txt')
print("sol 4=",e10_bow_vector)



def compute_likelihood(x, theta):
    return sum([xi * math.log(theta_i) for xi, theta_i in zip(x, theta)])

# Using the computed θ values for each language, compute the likelihood of e10.txt given each language
likelihood_e = compute_likelihood(e10_bow_vector, list(theta_e.values()))
likelihood_j = compute_likelihood(e10_bow_vector, list(theta_j.values()))
likelihood_s = compute_likelihood(e10_bow_vector, list(theta_s.values()))


print("Log-Likelihood for English:", likelihood_e)
print("Log-Likelihood for Japanese:", likelihood_j)
print("Log-Likelihood for Spanish:", likelihood_s)


def compute_posterior(likelihood, prior):
    return likelihood * prior  # since we are ignoring the marginal likelihood p(x)

# Compute the posterior probabilities for each language
posterior_e = compute_posterior(likelihood_e, prior_e)
posterior_j = compute_posterior(likelihood_j, prior_j)
posterior_s = compute_posterior(likelihood_s, prior_s)

# Print the results
print(f"p(y=e | x): {posterior_e}")
print(f"p(y=j | x): {posterior_j}")
print(f"p(y=s | x): {posterior_s}")

# Predict the class label of x
predicted_class = max([('e', posterior_e), ('j', posterior_j), ('s', posterior_s)], key=lambda x: x[1])[0]
print(f"Predicted class label of x: {predicted_class}")


# Define the characters we're interested in
characters = [chr(i) for i in range(97, 123)] + [' ']

# Define a function to compute the probability of a document given a language theta
def compute_document_probability(bow_vector, theta):
    prob = 0.0
    for xi, char in zip(bow_vector, characters):
        prob += xi * math.log(theta[char])
    return prob

# Define a function to classify a document based on computed probabilities
def classify_document(bow_vector):
    # Compute the posterior probabilities for each language
    prob_e = log_prior_e + compute_document_probability(bow_vector, theta_e)
    prob_j = log_prior_j + compute_document_probability(bow_vector, theta_j)
    prob_s = log_prior_s + compute_document_probability(bow_vector, theta_s)
    
    # Return the class with the highest probability
    return max([(prob_e, 'e'), (prob_j, 'j'), (prob_s, 's')], key=lambda x: x[0])[1]

# Load test documents
test_files = [f for f in os.listdir(DATA_PATH) if int(f[1:-4]) in range(10, 20)]

# Initialize confusion matrix
confusion_matrix = {
    'e': {'e': 0, 'j': 0, 's': 0},
    'j': {'e': 0, 'j': 0, 's': 0},
    's': {'e': 0, 'j': 0, 's': 0}
}

# For each test document, classify it and record the result in the confusion matrix
for file_name in test_files:
    true_class = file_name[0]
    bow_vector = generate_bow_vector(file_name)
    predicted_class = classify_document(bow_vector)
    confusion_matrix[true_class][predicted_class] += 1

# Print the confusion matrix
print("Confusion Matrix:\n")
print("True\Predicted\tEnglish\tJapanese\tSpanish")
print("English\t\t" + "\t".join(str(confusion_matrix['e'][cls]) for cls in ['e', 'j', 's']))
print("Japanese\t" + "\t".join(str(confusion_matrix['j'][cls]) for cls in ['e', 'j', 's']))
print("Spanish\t\t" + "\t".join(str(confusion_matrix['s'][cls]) for cls in ['e', 'j', 's']))
