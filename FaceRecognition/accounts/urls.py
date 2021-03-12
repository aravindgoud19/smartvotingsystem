from django.urls import path
from django.conf.urls import include, url

from . import views
from .models import user

urlpatterns = [
    path("register",views.register,name="register"),
    path("login",views.login,name="login"),
    path("adminlogin",views.adminlogin,name="adminlogin"),
    path("trainer",views.trainer,name="trainer"),
    path("vote",views.vote,name="vote"),
    url("conformvote",views.conformvote,name="conformvote"),
    path("elections",views.elections,name="elections"),
    path("viewresults",views.viewresults,name="viewresults"),
    path("error",views.error,name="error"),
    path("conform",views.conform,name="conform"),
    path("adminlogout",views.adminlogout,name="adminlogout"),
    path("pollingstatus",views.pollingstatus,name="pollingstatus")

]