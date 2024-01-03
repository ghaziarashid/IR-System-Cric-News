from flask import Flask, render_template, request
import pandas as pd
from rapidfuzz import fuzz, process
from flask import redirect
from flask import jsonify
from whoosh.fields import Schema, TEXT
from whoosh.index import create_in

# TfidfVectorizer (from scikit-learn)
from sklearn.feature_extraction.text import TfidfVectorizer

# Gensim
from gensim.corpora import Dictionary
import openai


app = Flask(__name__)
# Set your OpenAI API key
openai.api_key = "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"
    



# Add Document Route
@app.route('/add_document', methods=['POST'])
def add_document():
    global df  # Declare df as a global variable

    doc_name = request.form['doc_name']
    doc_content = request.form['doc_content']

    # Create a new DataFrame with the new row
    new_row = pd.DataFrame({'Document ID': [doc_name], 'Content': [doc_content]})
    
    # Concatenate the new DataFrame with the existing DataFrame
    df = pd.concat([df, new_row], ignore_index=True)

    # Save the DataFrame back to the CSV file
    df.to_csv('static/corpus.csv', index=False)

    return redirect('/admin')

# Update Document Route
@app.route('/update_document', methods=['POST'])
def update_document():
    update_doc_name = request.form['update_doc_name']

    # Find the row to update
    row_to_update = df[df['Document ID'] == update_doc_name]

    # Check if the document exists
    if row_to_update.empty:
        return render_template('no_document_found.html', action='update')

    # Render a template to allow the user to update the document content
    # (you can customize this template based on your requirements)
    return render_template('update_document.html', row_to_update=row_to_update)


# Update Document Form Submission Route
@app.route('/submit_update', methods=['POST'])
def submit_update():
    # Retrieve updated information from the form
    updated_doc_name = request.form['updated_doc_name']
    updated_doc_content = request.form['updated_doc_content']

    # Retrieve the original document name from the hidden input
    original_doc_name = request.form['original_doc_name']

    # Check if the document name is being updated
    if updated_doc_name != original_doc_name:
        # Update the name of the corresponding row in the DataFrame
        df.loc[df['Document ID'] == original_doc_name, 'Document ID'] = updated_doc_name

    # Update the content of the corresponding row in the DataFrame
    df.loc[df['Document ID'] == updated_doc_name, 'Content'] = updated_doc_content

    # Save the DataFrame back to the CSV file
    df.to_csv('static/corpus.csv', index=False)

    return redirect('/admin')



# Delete Document Route
@app.route('/delete_document', methods=['POST'])
def delete_document():
    delete_doc_name = request.form['delete_doc_name']

    # Check if the document exists
    if delete_doc_name not in df['Document ID'].values:
        return render_template('no_document_found.html', action='delete')

    # Drop the row with the specified document name
    df.drop(df[df['Document ID'] == delete_doc_name].index, inplace=True)

    # Save the DataFrame back to the CSV file
    df.to_csv('static/corpus.csv', index=False)

    return redirect('/admin')


@app.route('/ask', methods=['POST']) 
def ask():
    user_question = request.form['question']
    full_content = request.form['full_content']

    response = ask_chatbot(user_question, full_content)

    # Render the full_document.html template with the question, answer, and full content
    return render_template('full_document.html', question=user_question, answer=response, document="Document", full_content=full_content)


# ... (remaining code)

def ask_chatbot(question, context):
    prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100
    )
    generated_text = response['choices'][0]['text']
    return generated_text


# Load the CSV file into a DataFrame
df = pd.read_csv('static/corpus.csv')

# Function for fuzzy retrieval
def fuzzy_retrieval(user_query, documents, threshold=70):
    results = []

    for index, row in documents.iterrows():
        # Concatenate document name and content for matching
        document_text = f"{row['Document ID']} {row['Content']}"
        
        # Use rapidfuzz for fuzzy matching (handles typos)
        similarity_score = fuzz.partial_ratio(user_query, document_text)
        
        # Add the document and the relevant line to the results if above the threshold
        if similarity_score >= threshold:
            relevant_line = process.extractOne(user_query, document_text.split('\n'), scorer=fuzz.partial_ratio)[0]
            results.append((row['Document ID'], relevant_line, similarity_score))

    # Sort results by similarity score in descending order
    results.sort(key=lambda x: x[2], reverse=True)

    return results

# Define the route for the home page
@app.route('/')
def splash():
    return render_template('splash.html')


@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')


# Define the route for displaying the full document
@app.route('/full_document/<document>')
def full_document(document):
    # Retrieve the full content of the specified document
    full_content = df[df['Document ID'] == document]['Content'].iloc[0]

    return render_template('full_document.html', document=document, full_content=full_content)

# Define the route for processing the form data
@app.route('/result', methods=['POST'])
def result():
    user_query = request.form['query']
    
    # Perform fuzzy retrieval
    retrieval_results = fuzzy_retrieval(user_query, df)

    if not retrieval_results:
        result_message = "No match found."
    else:
        result_message = []
        for result in retrieval_results:
            document_name, relevant_line, similarity_score = result
            result_message.append({
                'document': document_name,
                'line': relevant_line,
                'score': similarity_score
            })

    return render_template('result.html', query=user_query, result=result_message)



if __name__ == '__main__':
    app.run(debug=True)
