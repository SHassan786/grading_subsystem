''' Simple flask app to load word2vec google news model from gensim to spacy and have grade api to return 
correctness of student response by comparing similarity of student response to golden answer'''

from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
# from bson import ObjectId
from dotenv import load_dotenv
import os
import spacy
import gensim.downloader as api
import string
import nltk
from nltk.corpus import stopwords
from rouge import Rouge


# dotenv_path = os.path.join(os.path.dirname(__file__), '../config.env')
# load_dotenv(dotenv_path)
# print(os.getenv('ATLAS_URI'))

app = Flask(__name__)
# app.config['MONGO_URI'] = os.getenv('ATLAS_URI')
# mongo = PyMongo(app)

# Load word2vec google news model from gensim to spacy in function
def load_word2vec():
    print('Loading word2vec model from gensim...')
    word2vec_model = api.load('word2vec-google-news-300')

    print('Loading word2vec model to spacy...')
    nlp = spacy.blank('en')
    vocab = nlp.vocab

    # Add vectors to spacy vocab
    for word, vector in zip(word2vec_model.index_to_key, word2vec_model.vectors):
        vocab.set_vector(word, vector)

    print('Model loaded.')

    print('Saving word2vec model to disk...')
    nlp.to_disk('./models/word2vec')
    
    return nlp

def load_saved_word2vec():
    print('Loading word2vec model from disk...')
    nlp = spacy.blank('en')
    nlp.from_disk('./models/word2vec')

    print('Model loaded.')
    
    return nlp

nlp = load_word2vec()
# nlp = load_saved_word2vec()
nltk.download('stopwords')  # Download the stopwords dataset if not already downloaded
stop_words = set(stopwords.words('english'))  # Get the list of English stop words

# Grade test API------ For testing purpose
@app.route('/test', methods=['POST'])
def grade():
    # Get student response and golden answer from request
    student_response = request.json['student_response']
    golden_answer = request.json['golden_answer']

    # Calculate similarity of student response to golden answer
    similarity = nlp(student_response).similarity(nlp(golden_answer))

    # Return correctness of student response
    if similarity > 0.8:
        return jsonify({'correctness': 'Correct', 'similarity': similarity})
    else:
        return jsonify({'correctness': 'Incorrect', 'similarity': similarity}) 


''' Api that takes quiz id from params, fetches quiz from mongodb, takes student id from request,
accesses questions from from quiz, accesses student response from quiz, grades student response
and stores correctness of student response in mongodb'''

# Grade student response API ------ No longer used in application
@app.route('/calculate_grade/<quiz_id>', methods=['POST'])
def calculate_grade(quiz_id):
    # Retrieve quiz from MongoDB
    # quiz = mongo.db.quiz.find_one({'_id': quiz_id})
    # dummy quiz
    quiz = {'_id': '1', 'title': 'Quiz 1', 'questions': ['1'], 
            'start_time': '2021-10-01T00:00:00Z', 'end_time': '2021-10-01T23:59:59Z',
            'is_active': True, 'is_released': True, 'class': '1'}
    if not quiz:
        return jsonify({'message': 'Quiz not found'}), 404

    # Retrieve student id from request
    student_id = request.json['student_id']
    # dummy student id
    # student_id = '1'

    # Retrieve student responses for the quiz questions
    responses = []
    for question_id in quiz['questions']:
        # question = mongo.db.question.find_one({'_id': question_id})
        # dummy question
        question = {'_id': question_id, 'question': 'Where is Karachi located?', 
                    'answer': 'Pakistan', 'true_grade': 1, 
                    'responses': [{'_id': '1', 'student_answer': 'Pakistan', 'grade': 0, 'student': '1'},
                                  {'_id': '2', 'student_answer': 'Physics', 'grade': 0, 'student': '2'}]}
        if question:
            for response in question['responses']:
                if response['student'] == student_id:
                    responses.append(response)
            
    # Calculate grades using Word2Vec model
    for response in responses:
        student_answer = response['student_answer']
        similarity_score = calculate_similarity(student_answer, question['answer'], question['question'], question['true_grade'])
        grade = calculate_grade(similarity_score, question['true_grade'])

        print(f'Grading response number {response["_id"]} by student number {response["student"]} {response["student_answer"]} with assigned grade {grade} and similarity score {similarity_score} to golden answer {question["answer"]}')

        # Update response with grade
        # mongo.db.question.update_one(
        #     {'_id': question['_id'], 'responses._id': response['_id']},
        #     {'$set': {'responses.$.grade': grade}}
        # )
        # Update student quiz_grade with grade
        # quiz_grade is array of quiz id and grade
        # quiz_grade = {'quiz_id': quiz_id, 'grade': grade}
        # mongo.db.student.update_one(
        #     {'_id': response['student']},
        #     {'$push': {'quiz_grades': quiz_grade}}
        # )

    return jsonify({'message': 'Grades calculated and updated successfully'})


''' Api that takes array of student response, golden answer, true grade and question from request,
grades each student response and returns grade of each student response'''

# Grade student response API
@app.route('/grade', methods=['POST'])
def grade_student_response():
    # Get student response and golden answer from request
    studentRes = request.json.get('student_res', [])
    quiz_id = request.json.get('quiz_id', '')
    student_id = request.json.get('student_id', '')

    for res in studentRes:
        student_response = res['student_answer']
        golden_answer = res['answer']
        true_grade = res['grade']
        question = res['question']

        # Calculate similarity of student response to golden answer
        similarity = calculate_similarity(student_response, golden_answer, question, true_grade)
        grade = calculate_grade(similarity, true_grade)
        similarity_rouge = calculate_rouge_similarity(student_response, golden_answer, true_grade)
        
        res['grade'] = grade
        res['similarity'] = similarity
        res['similarity_rouge'] = similarity_rouge

    return jsonify({'questions': studentRes})

def calculate_rouge_similarity(student_ans, gold_ans, marks):
    rouge = Rouge()
    scores = rouge.get_scores(student_ans, gold_ans)
    similarity = scores[0]['rouge-1']['f']
    # convert marks to float
    marks = float(marks)
    return similarity*marks

def preprocess_text(text):
    # Tokenization, lowercasing, and punctuation removal
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Tokenize by space
    tokens = text.split()
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]  # Define stop_words if needed
    return " ".join(tokens)

# Function to calculate sentence similarity using spaCy vectors
def calculate_similarity(student_ans, gold_ans, question,marks):
    question_word = preprocess_text(question).split()
    student_ansp = preprocess_text(student_ans)
    # print()
    # print('student_ansp:',student_ansp)
    # print('test', student_ansp not in question_word)
    gold_ansp = preprocess_text(gold_ans)
    doc1 = nlp(' '.join([word for word in student_ansp.split() if word not in question_word]))
    doc2 = nlp(' '.join([word for word in gold_ansp.split() if word not in question_word]))
    # print('doc1:',doc1)
    # print('doc2:',doc2)
    similarity1 = doc1.similarity(doc2)
    # print('similarity1:',similarity1)
    # convert marks to float
    marks = float(marks)
    return (similarity1*marks)

def calculate_grade(similarity_score, true_grade):
    true_grade = int(true_grade)
    if true_grade==1 and similarity_score > 0.63:
        return 1
    elif true_grade==2 and similarity_score > 1.26:
        return 2
    elif true_grade==2 and similarity_score > 0.63 and similarity_score<1.26:
        return 1
    return 0
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    