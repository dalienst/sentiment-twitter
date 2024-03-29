import pickle
from django.http import HttpResponseBadRequest
from django.shortcuts import render
from django.contrib import messages

from sentimental.forms import TweetForm

# loading the pickle model
try:
    with open("mlmodel/model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError("The pickle model file could not be found.")
except Exception as e:
    raise RuntimeError("An error occurred while loading the pickle model: {}".format(e))


def predict_sentiment(tweet):
    """
    Predicts the sentiment of a given tweet using the loaded model.
    """
    try:
        sentiment = model.predict([tweet])
        return sentiment[0]
    except Exception as e:
        raise RuntimeError("An error occurred while predicting sentiment: {}".format(e))


def predict_depression(request):
    if request.method == "POST":
        form = TweetForm(request.POST)
        if form.is_valid():
            tweet = form.cleaned_data["tweet"]
            sentiment = predict_sentiment(tweet)
            # sentiment_message = "Positive" if sentiment == 0 else "Negative" 
            if sentiment == 0:
                sentiment = "Negative"
            else:
                sentiment = "Positive"
            messages.success(request, f"Prediction result: {sentiment}")
            # Clear the form data
            form = TweetForm()
            return render(request, "index.html", {"form": form})
    else:
        form = TweetForm()
    return render(request, "index.html", {"form": form})
