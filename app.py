from flask import Flask, render_template, request
from assistant import get_scheme_answer

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():

    answer = ""

    if request.method == "POST":

        query = request.form["query"]
        income = request.form["income"]
        state = request.form["state"]
        education = request.form["education"]
        category = request.form["category"]

        # Call AI engine from assistant.py
        answer = get_scheme_answer(query, income, state, education, category)

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=False)
