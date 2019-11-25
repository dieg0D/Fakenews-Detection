from django import forms

from .models import FormNews

class SearchForm(forms.ModelForm):
    class Meta:
        model = FormNews
        fields = ('texto', 'titulo', 'veiculo')
