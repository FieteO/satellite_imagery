from flask import Flask, render_template, url_for, request, redirect
from flask_bootstrap import Bootstrap

import matplotlib
# matplotlib.use('TkAgg')

import os
import model

from pathlib import Path
import sys # added!


app = Flask(__name__, template_folder='template', static_folder='static')
Bootstrap(app)
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
#<h1>Predicted Classes: {{ result.object_classes }}</h1>
"""
Routes
"""
@app.route('/', methods=['GET','POST'])
def index():
    sys.path.append("../..")  # added!
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = os.path.join('static', uploaded_file.filename)
            image_path = uploaded_file.filename
            #uploaded_file.save(image_path)
            cls, plt = model.get_prediction(image_path)

            filename = 'plot.png'
            # plt.savefig(f'../{img_path}')
            plt.savefig(f'static/{filename}')
            # img_path = f'/usr/src/app/{img_path}'
            result = {
                'object_classes': cls,
                'image_path': image_path,
                'filename': filename
            }
            models = ['unet','segnet']
            return render_template('result.html', result = result, plot_name='new_plot', plot_url=filename, models=models)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug = os.getenv('DEBUG'))