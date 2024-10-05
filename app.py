from flask import Flask, render_template, request
import pickle
import numpy as np

# Load pre-trained models
model_dtree = pickle.load(open('dtreemodel.pkl', 'rb'))
model_naiveb = pickle.load(open('naivemodel.pkl', 'rb'))

app = Flask(__name__)

# Function to classify text using your models
def classify_text(input_text, model_dtree, model_naiveb):
    models = [
        ('Decision Tree', model_dtree),
        ('Naive Bayes Multinomial', model_naiveb)
    ]
    
    result = {}

    for name, model in models:
        if hasattr(model, "predict_proba"):
            # Directly pass the input_text if models can handle raw text
            probas = model.predict_proba([input_text])  # Assuming models can accept raw text
            # Get top 3 predictions
            top_3_indices = np.argsort(probas, axis=1)[:,-3:][0][::-1]
            top_3_predictions = [model.classes_[i] for i in top_3_indices]
            result[name] = top_3_predictions
            
    # Combine predictions from Decision Tree and Naive Bayes
    pred = [result['Decision Tree'][0], result['Naive Bayes Multinomial'][1], result['Naive Bayes Multinomial'][2]]
    
    return pred

# Home route to show form and process input
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        # Predict the results
        result = classify_text(input_text, model_dtree, model_naiveb)
        return render_template('index.html', input_text=input_text, result=result)
    return render_template('index.html', result=None)


