from flask import Flask, render_template, request, redirect, url_for


app=Flask(__name__)


@app.route('/')
def main_page():
    return render_template("IY_Home_page.html")


if __name__=="__main__":
    app.run(debug=True)