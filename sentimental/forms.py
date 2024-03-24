# forms.py

from django import forms


class TweetForm(forms.Form):
    tweet = forms.CharField(label="Enter your Text", widget=forms.Textarea)
