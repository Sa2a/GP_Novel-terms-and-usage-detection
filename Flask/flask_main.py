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
    # data.apply(lambda x: max(x.relevance_no_correction,x.relevance_with_correction), axis=1 )
    data["Relevance Score"] = data.apply(lambda row: max(row["Relevance before correction"],row["Relevance after correction"]),axis=1)
    conditions = [
        (data['Correct'] == True) & (data["Relevance before correction"] <= meaningTh),
        (data['Correct'] == True) & (data["Relevance before correction"] > meaningTh),
        (data['Correct'] != True) & (data["Relevance Score"] <= novelTh),
        (data['Correct'] != True) & (data["Relevance Score"]> novelTh)
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
    return render_template(index_file,**locals(),   tables_original=[originalData.to_html(classes='data')],
    tables_filter=enumerate([
        data.query('Label == "Novel"')[["Sentence","Index","Word","Relevance Score"]].to_html(classes='data'),
        data.query('Label ==  "Misspelled"')[["Sentence","Index","Word","Relevance Score"]].to_html(classes='data'),
        data.query('Label == "Meaning Change"')[["Sentence","Index","Word","Relevance Score"]].to_html(classes='data'),
        data.query('Label ==  "Meaning No Change"')[["Sentence","Index","Word","Relevance Score"]].to_html(classes='data'),
        # data[(data.Label=='Novel') or (data.Label == 'Misspelled')].to_html(classes='data'),
    ]),names =["Novel","Misspelled","Meaning Change","Meaning No Change"], titles=data.columns.values)

if __name__ == "__main__":
    # host= "0.0.0.0"
    app.run(debug=True,port=8000)
