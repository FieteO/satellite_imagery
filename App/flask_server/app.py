from flask import Flask, render_template, url_for, request, redirect
from flask_bootstrap import Bootstrap

import os
import model

app = Flask(__name__, template_folder='../Template')
Bootstrap(app)

"""
Routes
"""
@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = os.path.join('static', uploaded_file.filename)
            #uploaded_file.save(image_path)
            cls, plt = model.get_prediction(image_path)
            result = {
                'object_classes': cls,
                'image_path': image_path,

            }
            plt.savefig('../static/new_plot.png')
            return render_template('result.html', result = result,plot_name='new_plot', plot_url = '../static/new_plot.png')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug = True)