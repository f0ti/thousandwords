#!/usr/bin/env python3

import os
import json
import openai

from flask import Flask, request, jsonify, render_template, redirect, url_for
from caption_generator import infer

app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/result')
def result():
    
    story = request.args.get('story')

    return open('result.html').read().format(story)

# send request to api
@app.route('/generate', methods=['POST', 'GET'])
def generate(): 

    if request.method == "POST":
        pass    

    else:
        sentence = request.form['utter']

        prompt = "Write my a story: \n" + sentence

        openai.api_key = os.getenv("OPENAI_API_KEY")

        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            temperature=0.69,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        response = json.loads(str(response))
        
        print(prompt)
        print(response)

        story = "%s, %s" % (sentence, response.get('choices')[0].get('text'))

        return redirect(url_for('result', story=story))

    return render_template('generate.html')

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(debug=False, threaded=True, port=5000)

