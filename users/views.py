from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from users.forms import UserForm

#이용약관
from django.utils.decorators import method_decorator
from .decorators import *
from django.contrib.auth.models import User
from django.views.generic import View
from django.contrib import messages

@method_decorator(logout_message_required, name='dispatch')
class AgreementView(View):
    def get(self, request, *args, **kwargs):
        request.session['agreement'] = False
        return render(request, 'users/agreement.html')

    def post(self, request, *args, **kwarg):
        if request.POST.get('agreement1', False) and request.POST.get('agreement2', False):
            request.session['agreement'] = True
            return redirect('/users/signup/')
        else:
            messages.info(request, "약관에 모두 동의해주세요.")
            return render(request, 'users/agreement.html')

#회원가입
def signup(request):
    if request.method == "POST":
        form = UserForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)  # 사용자 인증
            login(request, user)  # 로그인
            return redirect('/')
    else:
        form = UserForm()
    return render(request, 'users/signup.html', {'form': form})

def my_view(request):
    if request.method == 'POST':
        form = UserForm(request, request.POST)
        if form.is_valid():
            login(request, form.get_user())
            return redirect(request.GET.get('next') or 'App/index.html')

def delete(request, pk):
    if request.user.is_authenticated:
        article = get_object_or_404(ARticle, pk=pk)
        article.delete()
    return redirect('App/index.html')