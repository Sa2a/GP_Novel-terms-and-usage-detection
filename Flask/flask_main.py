from flask import Flask, request, render_template
import sys, os
# from pathlib import Path
# # parent_direcory = Path(Path.cwd()).parent
# # sys.path.append(os.path.join(os.path.dirname(__file__)))
current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_directory))
# import Common
from Scanner import scan_text
app= Flask(__name__,template_folder=current_directory)





index_file ="index.html"
@app.route('/')
def homepage():
    return render_template(index_file,**locals())
@app.route('/predict', methods = ['POST','GET'])
def predict():
    text=request.form['texta']
    data= scan_text(text)
    return render_template(index_file,**locals(),   tables=[data.to_html(classes='data')], titles=data.columns.values)
        
if __name__ == "__main__":
    app.run(debug=True,port=8000, host= "0.0.0.0")
