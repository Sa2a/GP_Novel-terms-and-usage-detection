from flask import Flask, request, render_template
import sys, os
# from pathlib import Path
# # parent_direcory = Path(Path.cwd()).parent
# # sys.path.append(os.path.join(os.path.dirname(__file__)))
current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_directory))
# import Common
import numpy as np
from Scanner import scan_text
app= Flask(__name__,template_folder="./grad website/startbootstrap-agency-gh-pages",static_folder="./grad website/startbootstrap-agency-gh-pages")

originalData = []
# originalData = []
def applyFilter(novelTh,meaningTh):
    global originalData
    data  = originalData.copy()
    conditions = [
        (data['Correct'] == True) & (data["Relevance before correction"] <= meaningTh),
        (data['Correct'] == True) & (data["Relevance before correction"] > meaningTh),
        (data['Correct'] != True) & (data[["Relevance before correction","Relevance after correction"]].max(axis=1) <= novelTh),
        (data['Correct'] != True) & (data[["Relevance before correction","Relevance after correction"]].max(axis=1) > novelTh)
        ]
    values = ['Meaning Change', 'Meaning No Change', 'Novel', 'Misspelled']
    # create a new column and use np.select to assign values to it using our lists as arguments
    data['Label'] = np.select(conditions, values)
    return data



index_file ="index.html"
@app.route('/')
def homepage():
    return render_template(index_file,**locals())
@app.route('/predict', methods = ['POST','GET'])
def predict():
    text=request.form['texta']
    global originalData
    originalData= scan_text(text)
    return render_template(index_file,**locals(),   tables_original=[originalData.to_html(classes='data')], titles=originalData.columns.values)

@app.route('/predict/filter', methods = ['POST','GET'])
def filter():
    novelThreshold=np.float64(request.form['novel_threshold'])
    meaningThreshold=np.float64(request.form['meaning_threshold'])
    print(type(meaningThreshold))
    data  = applyFilter(novelThreshold,meaningThreshold)
    return render_template(index_file,**locals(),   
    tables_filter=[
        data.query('Label ==  "Meaning No Change"').to_html(classes='data'),
        data.query('Label == "Meaning Change"').to_html(classes='data'),
        data.query('Label == "Novel"').to_html(classes='data'),
        data.query('Label ==  "Misspelled"').to_html(classes='data')
        # data[(data.Label=='Novel') or (data.Label == 'Misspelled')].to_html(classes='data'),
    ],names =["Meaning No Change","Meaning Change","Novel","Misspelled"], titles=data.columns.values)

if __name__ == "__main__":
    app.run(debug=True,port=8000, host= "0.0.0.0")
