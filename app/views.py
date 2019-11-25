from django.shortcuts import render
from django.shortcuts import redirect
from django.shortcuts import get_object_or_404
from .forms import SearchForm
from .models import FormNews

# Create your views here.

def news(request):
    if(request.method == 'POST'):
        form = SearchForm(request.POST)
        if form.is_valid:
            news = form.save(commit=False)
            news.save()
            return redirect('success', pk=news.pk)
    else:
        form = SearchForm()
        dict = {'form': form}
    return render(request, 'form.html', context=dict)

def success(request, pk):
    form = get_object_or_404(FormNews, pk=pk)
    dict = {'form': form}
    return render(request, 'success.html', dict)