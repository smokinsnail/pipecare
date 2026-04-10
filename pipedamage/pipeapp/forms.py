from django import forms

class ImageForm(forms.Form):
    imageFile = forms.FileField(label='Select a Video to Upload',widget=forms.ClearableFileInput(attrs={'id':'picker'}))
    #imageFile = forms.FileField(label='Select a Video to Upload',widget=forms.ClearableFileInput())