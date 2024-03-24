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
    # Handle the case where the pickle file is not found
    raise FileNotFoundError("The pickle model file could not be found.")
except Exception as e:
    # Handle other exceptions, such as unpickling errors
    raise RuntimeError("An error occurred while loading the pickle model: {}".format(e))


def predict_sentiment(tweet):
    """
    Predicts the sentiment of a given tweet using the loaded model.
    """
    try:
        sentiment = model.predict([tweet])
        return sentiment[0]
    except Exception as e:
        # Handle prediction errors
        raise RuntimeError("An error occurred while predicting sentiment: {}".format(e))


def predict_depression(request):
    if request.method == "POST":
        form = TweetForm(request.POST)
        if form.is_valid():
            tweet = form.cleaned_data["tweet"]
            sentiment = predict_sentiment(tweet)
            context = {"prediction": sentiment}
            return render(request, "index.html", context)
        else:
            # Form is not valid, render it again with error messages
            return HttpResponseBadRequest("Invalid form data. Please check your input.")
    else:
        form = TweetForm()
    return render(request, "index.html", {"form": form})
