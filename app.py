from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from model import predict_image, class_names  

app = Flask(__name__)

# Create a folder for uploads
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        
        # Save the uploaded file securely
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Predict the terrain type using your pre-trained model
        predicted_class_index = predict_image(file_path)  # Call the function from model.py
        predicted_class = class_names[predicted_class_index]

        # Redirect to results page
        return redirect(url_for('prediction_result', predicted_class=predicted_class, filename=filename))
    
    return render_template('upload.html')

@app.route('/result')
def prediction_result():
    predicted_class = request.args.get('predicted_class')
    filename = request.args.get('filename')
    img_url = url_for('uploaded_file', filename=filename)
    return render_template('result.html', predicted_class=predicted_class, img_url=img_url)

if __name__ == '__main__':
    app.run(debug=True)
